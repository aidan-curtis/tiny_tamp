from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import trimesh
from trimesh.points import plane_transform
from trimesh.ray.ray_triangle import RayMeshIntersector

import tiny_tamp.pb_utils as pbu
from tiny_tamp.motion_planning.motion_planners.rrt_connect import birrt
from tiny_tamp.structs import (
    ARM_GROUP,
    COLLISION_DISTANCE,
    COLLISION_EPSILON,
    PANDA_IGNORE_COLLISIONS,
    SELF_COLLISIONS,
    ActivateGrasp,
    Attachment,
    Conf,
    DeactivateGrasp,
    Grasp,
    Sequence,
    SimulatorInstance,
    Trajectory,
    WorldBelief,
)


def get_plan_motion_fn(
    sim: SimulatorInstance, environment: List[int] = [], debug: bool = False
):
    """Plan a motion from one configuration to another."""

    def fn(q1: Conf, q2: Conf, attachments: List[Attachment] = []):
        obstacles = list(environment)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign(sim)

        resolutions = math.radians(10) * np.ones(len(q2.joints))

        path = plan_joint_motion(
            sim,
            q2.joints,
            q2.positions,
            resolutions=resolutions,
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=SELF_COLLISIONS,
            max_distance=COLLISION_DISTANCE,
            restarts=1,
            iterations=5,
            smooth=100,
            debug=debug,
            disable_collisions=False,
        )

        if path is None:
            return None

        return Trajectory(sim.robot, q2.joints, path, attachments=attachments)

    return fn


def solve_ik(sim: SimulatorInstance, link, target_pose, start_q=None, obstacles=[]):
    randomize_seed = start_q is None
    max_attempts = 100
    arm_joints = sim.get_group_joints(ARM_GROUP)
    ranges = [
        pbu.get_joint_limits(sim.robot, joint, client=sim.client)
        for joint in arm_joints
    ]

    for i in range(max_attempts):
        # Start with the current joint positions and then randomize within limits after
        if not randomize_seed:
            initialization_sample = start_q
            randomize_seed = True
        else:
            initialization_sample = [random.uniform(r.lower, r.upper) for r in ranges]

        pbu.set_joint_positions(
            sim.robot, arm_joints, initialization_sample, client=sim.client
        )

        conf = sim.client.calculateInverseKinematics(
            int(sim.robot),
            link,
            target_pose[0],
            target_pose[1],
            residualThreshold=0.00001,
            maxNumIterations=5000,
        )

        # Need to extract the arm component of the returned joints
        conf = [
            q
            for q, j in zip(conf, pbu.get_movable_joints(sim.robot, client=sim.client))
            if j in arm_joints
        ]

        lower, upper = list(zip(*[(r.lower, r.upper) for r in ranges]))

        assert len(arm_joints) == len(conf)
        pbu.set_joint_positions(sim.robot, arm_joints, conf, client=sim.client)

        if not pbu.all_between(lower, conf, upper):
            print("IK solution outside limits")
            continue

        contact_points = []
        for obstacle in obstacles:
            contact_points += sim.client.getClosestPoints(
                bodyA=obstacle, bodyB=sim.robot, distance=pbu.MAX_DISTANCE
            )

        all_joints = pbu.get_joints(sim.robot, client=sim.client)
        check_link_pairs = pbu.get_self_link_pairs(
            sim.robot, all_joints, PANDA_IGNORE_COLLISIONS, client=sim.client
        )

        self_collision = False
        for link1, link2 in check_link_pairs:
            if pbu.pairwise_link_collision(
                sim.robot, link1, sim.robot, link2, client=sim.client
            ):
                print(link1, link2)
                self_collision = True

        if self_collision:
            print("Self collision")
            continue

        # Print contact points if there are any
        if contact_points:
            print("Collision!")
            continue

        pose = pbu.get_link_pose(sim.robot, link, client=sim.client)
        trans_diff, rot_diff = pbu.get_pose_distance(target_pose, pose)

        if trans_diff < 0.001 and rot_diff < 0.01:
            return list(conf)
        else:
            print("IK Error: {}, {}".format(trans_diff, rot_diff))

    return None


def plan_workspace_motion(
    sim: SimulatorInstance,
    tool_waypoints: List[pbu.Pose],
    attachment: Attachment = None,
    obstacles: List[int] = [],
    max_attempts=2,
    debug=False,
) -> List[List[float]]:
    """Return a joint path that moves the tool along the tool waypoints.

    This is useful if you want to move the gripper in a straight line
    path.
    """
    assert tool_waypoints

    tool_link = sim.tool_link
    parts = [sim.robot] + ([] if attachment is None else [attachment.child])
    arm_joints = sim.get_group_joints(ARM_GROUP)

    collision_fn = get_collision_fn(
        sim,
        arm_joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=SELF_COLLISIONS,
    )

    for attempts in range(max_attempts):
        arm_conf = solve_ik(sim, tool_link, tool_waypoints[0])

        if arm_conf is None or collision_fn(arm_conf):
            continue

        arm_waypoints = [arm_conf]
        for tool_pose in tool_waypoints[1:]:
            arm_conf = solve_ik(sim, tool_link, tool_pose, start_q=arm_waypoints[-1])
            if arm_conf is None or collision_fn(arm_conf):
                break

            arm_waypoints.append(arm_conf)
        else:
            pbu.set_joint_positions(
                sim.robot, arm_joints, arm_waypoints[-1], client=sim.client
            )
            if attachment is not None:
                attachment.assign(sim)
            if any(
                pbu.pairwise_collisions(
                    part,
                    obstacles,
                    max_distance=(COLLISION_DISTANCE + COLLISION_EPSILON),
                    client=sim.client,
                )
                for part in parts
            ):
                if debug:
                    pbu.wait_if_gui(client=sim.client)
                continue
            arm_path = pbu.interpolate_joint_waypoints(
                sim.robot, arm_joints, arm_waypoints, client=sim.client
            )

            if any(collision_fn(q) for q in arm_path):
                if debug:
                    pbu.wait_if_gui(client=sim.client)
                continue

            print(
                "Found path with {} waypoints and {} configurations after {} attempts".format(
                    len(arm_waypoints), len(arm_path), attempts + 1
                )
            )
            print(arm_path)

            return arm_path
    return None


def fixed_grasp_sampler(
    sim: SimulatorInstance, belief: WorldBelief
) -> Callable[[int], Grasp]:
    def gen_fn(obj: int) -> Grasp:
        closed_conf, _ = sim.get_group_limits(sim.gripper_group)
        closed_position = closed_conf[0] * (1 + 5e-2)
        grasp_pose = pbu.multiply(
            pbu.Pose(euler=pbu.Euler(pitch=-np.pi / 2.0)),
            pbu.Pose(pbu.Point(z=-0.01)),
        )
        return Grasp(
            attachment=Attachment(sim.robot, sim.tool_link, obj, grasp_pose),
            closed_position=closed_position,
        )

    return gen_fn


def compute_gripper_path(pose: pbu.Pose, grasp: Grasp, pos_step_size=0.02):
    grasp_pose = pbu.multiply(pose, pbu.invert(grasp.attachment.parent_T_child))
    pregrasp_pose = pbu.multiply(
        pose, pbu.invert(grasp.get_pregrasp_pose(grasp.attachment.parent_T_child))
    )
    gripper_path = list(
        pbu.interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    return gripper_path


def workspace_collision(
    sim: SimulatorInstance,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
):
    gripper = sim.get_component(sim.gripper_group)

    if open_gripper:
        _, open_conf = sim.get_group_limits(sim.gripper_group)
        gripper_joints = sim.get_component_joints(sim.gripper_group)
        pbu.set_joint_positions(gripper, gripper_joints, open_conf, client=sim.client)

    parent_link = sim.get_group_parent(sim.gripper_group)
    parent_from_tool = pbu.get_relative_pose(
        sim.robot, sim.tool_link, parent_link, client=sim.client
    )

    parts = [gripper]
    if grasp is not None:
        parts.append(grasp.body)

    for i, gripper_pose in enumerate(gripper_path):
        pbu.set_pose(
            gripper,
            pbu.multiply(gripper_pose, pbu.invert(parent_from_tool)),
            client=sim.client,
        )
        if grasp is not None:
            pbu.set_pose(
                grasp.body, pbu.multiply(gripper_pose, grasp.value), client=sim.client
            )

        distance = (
            (COLLISION_DISTANCE + COLLISION_EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )
        if any(
            pbu.pairwise_collisions(
                part, obstacles, max_distance=distance, client=sim.client
            )
            for part in parts
        ):
            print("[workspace_collision] gripper path in collision")
            return True

    return False


def plan_prehensile(
    sim: SimulatorInstance,
    obj: int,
    pose: pbu.Pose,
    grasp: Grasp,
    environment: List[int] = [],
    debug: bool = False,
):
    pbu.set_pose(obj, pose, client=sim.client)
    gripper_path = compute_gripper_path(pose, grasp)
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(sim, gripper_waypoints, grasp=None, obstacles=environment):
        print("[plan_prehensile] gripper path in collision")
        return None

    arm_path = plan_workspace_motion(
        sim,
        gripper_waypoints,
        attachment=None,
        obstacles=environment,
        debug=debug,
    )

    if arm_path is None:
        print("[plan_prehensile] arm path none")

    return arm_path


def get_plan_pick_fn(sim: SimulatorInstance, environment: List[int] = [], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):
        obstacles = list(set(environment) - {obj})
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=obstacles, debug=debug
        )

        if arm_path is None:
            print("[get_plan_pick_fn] failed to find a plan")
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.group_joints[sim.arm_group],
            arm_path[::-1],
        )
        arm_conf = Conf(sim.robot, sim.group_joints[sim.arm_group], arm_traj.path[0])
        switch = ActivateGrasp(sim.robot, sim.tool_link, obj)
        reverse_traj = copy.deepcopy(arm_traj).reverse()
        reverse_traj.attachments = [grasp.attachment]
        commands = [arm_traj, switch, reverse_traj]
        sequence = Sequence(commands=commands, name="pick-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(sim: SimulatorInstance, environment=[], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):
        obstacles = list(set(environment) - {obj})
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=obstacles, debug=debug
        )
        if arm_path is None:
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.group_joints[sim.arm_group],
            arm_path[::-1],
            attachments=[grasp.attachment],
        )
        arm_conf = Conf(sim.robot, sim.group_joints[sim.arm_group], arm_traj.path[0])
        switch = DeactivateGrasp(sim.robot, sim.tool_link, obj)
        reverse_traj = copy.deepcopy(arm_traj).reverse()
        reverse_traj.attachments = []
        commands = [arm_traj, switch, reverse_traj]
        sequence = Sequence(commands=commands, name="place-{}".format(obj))

        return (arm_conf, sequence)

    return fn


def get_random_placement_pose(obj, surface_aabb, client):
    return pbu.Pose(
        point=pbu.Point(
            x=random.uniform(surface_aabb.lower[0], surface_aabb.upper[0]),
            y=random.uniform(surface_aabb.upper[1], surface_aabb.upper[1]),
            z=pbu.stable_z_on_aabb(obj, surface_aabb, client=client),
        ),
        euler=pbu.Euler(yaw=random.uniform(0, math.pi * 2)),
    )


def get_pick_place_plan(
    sim: SimulatorInstance,
    belief: WorldBelief,
    obj: int,
    grasp_sampler: Callable[[int, pbu.AABB, pbu.Pose], Grasp],
    motion_planner: Callable[[Conf, Conf, List[Attachment]], Trajectory],
    max_grasp_attempts=5,
    max_pick_attempts=5,
    max_place_attempts=5,
    placement_location=None,
) -> Sequence:
    # Only plan with the digital twin
    assert not sim.real_robot

    # Make sure the simulator matches the belief
    sim.set_belief(belief)

    body_saver = pbu.WorldSaver(client=sim.client)

    obstacles = sim.movable_objects + [sim.table]
    obj_pose = pbu.get_pose(obj, client=sim.client)

    pick_planner = get_plan_pick_fn(sim, environment=obstacles)
    place_planner = get_plan_place_fn(sim, environment=obstacles)

    statistics = {}
    q1 = Conf(
        sim.robot,
        sim.group_joints[sim.arm_group],
        sim.get_group_positions(sim.arm_group),
    )

    for gi in range(max_grasp_attempts):
        print("[Planner] grasp attempt " + str(gi))
        body_saver.restore()
        grasp = grasp_sampler(obj)

        print("[Planner] finding pick plan for grasp " + str(grasp))
        for _ in range(max_pick_attempts):
            body_saver.restore()
            pick = pick_planner(obj, obj_pose, grasp)
            if pick is not None:
                break

        if pick is None:
            continue

        q2, at1 = pick
        q2.assign(sim)

        for _ in range(max_place_attempts):
            if placement_location is not None:
                placement_pose = placement_location
            else:
                placement_pose = get_random_placement_pose(
                    obj, pbu.get_aabb(sim.table, client=sim.client), sim.client
                )
            body_saver.restore()
            place = place_planner(obj, placement_pose, grasp)

            if place is not None:
                break

        if place is None:
            continue

        q3, at2 = place
        q3.assign(sim)

        print("[Planner] finding pick motion plan")
        body_saver.restore()
        motion_plan1 = motion_planner(q1, q2)
        if motion_plan1 is None:
            continue

        print("[Planner] finding place motion plan")
        body_saver.restore()
        motion_plan2 = motion_planner(q2, q3, attachments=[grasp.attachment])
        if motion_plan2 is None:
            continue

        return (
            Sequence([motion_plan1, at1, motion_plan2, at2], name=f"pick-place({obj})"),
            statistics,
        )
    return None, statistics


#################Motion Planner######################


def get_collision_fn(
    sim: SimulatorInstance,
    joints: List[int],
    obstacles: List[int] = [],
    attachments: List[Attachment] = [],
    self_collisions: bool = True,
    disabled_collisions=set(),
    custom_limits={},
    use_aabb=False,
    cache=False,
    max_distance=pbu.MAX_DISTANCE,
    extra_collisions=None,
):
    check_link_pairs = (
        pbu.get_self_link_pairs(
            sim.robot, joints, disabled_collisions, client=sim.client
        )
        if self_collisions
        else []
    )
    moving_links = frozenset(
        link
        for link in pbu.get_moving_links(sim.robot, joints, client=sim.client)
        if pbu.can_collide(sim.robot, link, client=sim.client)
    )
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [pbu.CollisionPair(sim.robot, moving_links)] + list(
        map(pbu.parse_body, attached_bodies)
    )

    get_obstacle_aabb = pbu.cached_fn(
        pbu.get_buffered_aabb,
        cache=cache,
        max_distance=max_distance / 2.0,
        client=sim.client,
    )
    limits_fn = pbu.get_limits_fn(
        sim.robot, joints, custom_limits=custom_limits, client=sim.client
    )

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True

        pbu.set_joint_positions(sim.robot, joints, q, client=sim.client)

        for attachment in attachments:
            world_T_child = pbu.multiply(
                pbu.get_link_pose(
                    attachment.parent, attachment.parent_link, client=sim.client
                ),
                attachment.parent_T_child,
            )
            pbu.set_pose(attachment.child, world_T_child, client=sim.client)

            # Check if the attachment is in collision with objects in the environment
            if any(
                pbu.pairwise_collision(attachment.child, body, client=sim.client)
                for body in obstacles
            ):
                return True

        if extra_collisions is not None and extra_collisions(client=sim.client):
            return True

        get_moving_aabb = pbu.cached_fn(
            pbu.get_buffered_aabb,
            cache=True,
            max_distance=max_distance / 2.0,
            client=sim.client,
        )

        for link1, link2 in check_link_pairs:
            if (
                not use_aabb
                or pbu.aabb_overlap(
                    get_moving_aabb(sim.robot), get_moving_aabb(sim.robot)
                )
            ) and pbu.pairwise_link_collision(
                sim.robot, link1, sim.robot, link2, client=sim.client
            ):
                print("Link on link collision")
                print(sim.robot, link1, sim.robot, link2)
                return True

        for body1, body2 in itertools.product(moving_bodies, obstacles):
            if (
                not use_aabb
                or pbu.aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pbu.pairwise_collision(body1, body2, client=sim.client):
                print("Body on body collision")
                print(body1, body2)
                return True
        return False

    return collision_fn


def check_initial_end(
    sim: SimulatorInstance,
    joints: List[int],
    start_conf: List[float],
    end_conf: List[float],
    collision_fn: Callable[[List[float]], bool],
    debug=False,
    verbose=True,
):
    if collision_fn(start_conf, verbose=verbose):
        print("Warning: initial configuration is in collision {}".format(start_conf))
        if debug:
            pbu.set_joint_positions(sim.robot, joints, start_conf, client=sim.client)
            pbu.wait_if_gui(client=sim.client)
        return False
    if collision_fn(end_conf, verbose=verbose):
        print("Warning: end configuration is in collision {}".format(end_conf))
        if debug:
            pbu.set_joint_positions(sim.robot, joints, end_conf, client=sim.client)
            pbu.wait_if_gui(client=sim.client)
        return False
    return True


def plan_joint_motion(
    sim: SimulatorInstance,
    joints: List[int],
    end_conf: List[float],
    obstacles: List[int] = [],
    attachments: List[Attachment] = [],
    self_collisions: bool = True,
    disabled_collisions=set(),
    weights=None,
    resolutions=None,
    max_distance=pbu.MAX_DISTANCE,
    use_aabb=False,
    cache=True,
    custom_limits={},
    extra_collisions=None,
    debug=False,
    **kwargs,
):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = pbu.get_sample_fn(
        sim.robot, joints, custom_limits=custom_limits, client=sim.client
    )
    distance_fn = pbu.get_distance_fn(
        sim.robot, joints, weights=weights, client=sim.client
    )
    extend_fn = pbu.get_extend_fn(
        sim.robot, joints, resolutions=resolutions, client=sim.client
    )
    collision_fn = get_collision_fn(
        sim,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        use_aabb=use_aabb,
        cache=cache,
        extra_collisions=extra_collisions,
    )

    start_conf = pbu.get_joint_positions(sim.robot, joints, client=sim.client)
    if not check_initial_end(
        sim, joints, start_conf, end_conf, collision_fn, debug=debug
    ):
        return None

    return birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        client=sim.client,
        **kwargs,
    )


#### Antipodal Grasps ####
@dataclass
class ScoredGrasp:
    pose: pbu.Pose
    contact1: pbu.Point
    contact2: pbu.Point
    score: float


@dataclass
class Plane:
    normal: np.ndarray
    origin: np.ndarray


def mesh_from_obj(sim: SimulatorInstance, obj: int) -> pbu.Mesh:
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    [data] = pbu.get_visual_data(obj, -1, client=sim.client)
    filename = pbu.get_data_filename(data)
    scale = pbu.get_data_scale(data)
    if filename == pbu.UNKNOWN_FILE:
        raise RuntimeError(filename)
    elif filename == "":
        # Unknown mesh, approximate with bounding box
        aabb = pbu.get_aabb(obj, client=sim.client)
        aabb_center = pbu.get_aabb_center(aabb)
        centered_aabb = pbu.AABB(
            lower=aabb.lower - aabb_center, upper=aabb.upper - aabb_center
        )
        mesh = pbu.mesh_from_points(pbu.get_aabb_vertices(centered_aabb))
    else:
        mesh = pbu.read_obj(filename, decompose=False)

    vertices = [scale * np.array(vertex) for vertex in mesh.vertices]
    vertices = pbu.tform_points(pbu.get_data_pose(data), vertices)
    return pbu.Mesh(vertices, mesh.faces)


def extract_normal(mesh, index):
    return np.array(mesh.face_normals[index, :])


def point_plane_distance(plane: Plane, point, signed=True):
    signed_distance = np.dot(plane.normal, np.array(point) - np.array(plane.origin))
    if signed:
        return signed_distance
    return abs(signed_distance)


def project_plane(plane: Plane, point):
    return np.array(point) - point_plane_distance(plane, point) * plane.normal


def get_plane_quat(normal):
    plane = Plane(normal, np.zeros(3))
    tform = np.linalg.inv(plane_transform(plane.origin, -normal))  # origin=None
    quat1 = pbu.quat_from_matrix(tform)
    pose1 = pbu.Pose(plane.origin, euler=pbu.euler_from_quat(quat1))

    projection_world = project_plane(plane, np.array([0, 0, 1]))
    projection = pbu.tform_point(pbu.invert(pose1), projection_world)
    yaw = pbu.get_yaw(projection[:2])
    quat2 = pbu.multiply_quats(quat1, pbu.quat_from_euler(pbu.Euler(yaw=yaw)))

    return quat2


def sample_grasp(
    obj,
    point1,
    point2,
    pitches=[-np.pi, np.pi],
    discrete_pitch=False,
    finger_length=0,
    **kwargs,
):
    grasp_point = pbu.convex_combination(point1, point2)
    direction2 = point2 - point1
    quat = get_plane_quat(direction2)  # Biases toward the smallest rotation to align
    pitches = sorted(pitches)

    while True:
        if discrete_pitch:
            pitch = random.choice(pitches)
        else:
            pitch_range = [pitches[0], pitches[-1]]
            pitch = random.uniform(*pitch_range)
        roll = random.choice([0, np.pi])

        grasp_quat = pbu.multiply_quats(
            quat,
            pbu.quat_from_euler(pbu.Euler(roll=np.pi / 2)),
            pbu.quat_from_euler(pbu.Euler(pitch=np.pi + pitch)),
            pbu.quat_from_euler(pbu.Euler(roll=roll)),
        )
        grasp_pose = pbu.Pose(grasp_point, pbu.euler_from_quat(grasp_quat))
        grasp_pose = pbu.multiply(grasp_pose, pbu.Pose(pbu.Point(x=finger_length)))

        yield pbu.invert(grasp_pose), []


def tuplify_score(s):
    if isinstance(s, tuple):
        return s
    return (s,)


def negate_score(s):
    if isinstance(s, tuple):
        return s.__class__(map(negate_score, s))
    return -s


def combine_scores(score, *scores):
    combined_score = tuplify_score(score)
    for other_score in scores:
        combined_score = combined_score + tuplify_score(other_score)
    return combined_score


def score_torque(mesh, tool_from_grasp, **kwargs):
    center_mass = mesh.center_mass
    x, _, z = pbu.tform_point(tool_from_grasp, center_mass)  # Distance in xz plane
    return -pbu.get_length([x, z])


def sample_sphere_surface(d, uniform=True):
    while True:
        v = np.random.randn(d)
        r = np.sqrt(v.dot(v))
        if not uniform or (r <= 1.0):
            return v / r


def score_overlap(
    intersector,
    point1,
    point2,
    num_samples=15,
    radius=1.5e-2,
    draw=False,
    verbose=False,
    **kwargs,
):
    handles = []
    if draw:
        handles.append(pbu.add_line(point1, point2, color=pbu.RED))
    midpoint = np.average([point1, point2], axis=0)
    direction1 = point1 - point2
    direction2 = point2 - point1

    origins = []
    for _ in range(num_samples):
        other_direction = radius * sample_sphere_surface(d=3)
        orthogonal_direction = np.cross(
            pbu.get_unit_vector(direction1), other_direction
        )
        orthogonal_direction = radius * pbu.get_unit_vector(orthogonal_direction)
        origin = midpoint + orthogonal_direction
        origins.append(origin)
        if draw:
            handles.append(pbu.add_line(midpoint, origin, color=pbu.RED))
    rays = list(range(len(origins)))

    direction_differences = []
    for direction in [direction1, direction2]:
        point = midpoint + direction / 2.0
        contact_distance = pbu.get_distance(midpoint, point)
        results = intersector.intersects_id(
            origins,
            len(origins) * [direction],
            return_locations=True,
            multiple_hits=True,
        )
        intersections_from_ray = {}
        for face, ray, location in zip(*results):
            intersections_from_ray.setdefault(ray, []).append((face, location))

        differences = []
        for ray in rays:
            if ray in intersections_from_ray:
                face, location = min(
                    intersections_from_ray[ray],
                    key=lambda pair: pbu.get_distance(point, pair[-1]),
                )
                distance = pbu.get_distance(origins[ray], location)
                difference = abs(contact_distance - distance)
            else:
                difference = np.nan  # INF
            differences.append(difference)
        direction_differences.append(differences)

    differences1, differences2 = direction_differences
    combined = differences1 + differences2
    percent = np.count_nonzero(~np.isnan(combined)) / (len(combined))
    np.nanmean(combined)

    score = percent

    if verbose:
        print(
            "Score: {} | Percent1: {} | Average1: {:.3f} | Percent2: {} | Average2: {:.3f}".format(
                score,
                np.mean(~np.isnan(differences1)),
                np.nanmean(differences1),
                np.mean(~np.isnan(differences2)),
                np.nanmean(differences2),
            )
        )
    if draw:
        pbu.wait_if_gui()
        pbu.remove_handles(handles, **kwargs)
    return score


def antipodal_grasp_sampler(
    sim: SimulatorInstance,
    belief: WorldBelief,
    max_width=np.inf,
    target_tolerance=np.pi / 4,
    antipodal_tolerance=np.pi / 16.0,
    z_threshold=-np.inf,
    max_attempts=np.inf,
) -> Callable[[int], Grasp]:
    def gen_fn(obj: int) -> Grasp:
        target_vector = pbu.get_unit_vector(np.array([0, 0, 1]))

        pb_mesh = mesh_from_obj(sim, obj)
        # handles = draw_mesh(Mesh(vertices, faces))

        mesh = trimesh.Trimesh(pb_mesh.vertices, pb_mesh.faces)
        mesh.fix_normals()

        aabb = pbu.AABB(*mesh.bounds)
        surface_z = aabb.lower[2]
        min_z = surface_z + z_threshold
        intersector = RayMeshIntersector(mesh)

        attempts = last_attempts = 0

        while attempts < max_attempts:
            attempts += 1
            last_attempts += 1

            [point1, point2], [index1, index2] = trimesh.sample.sample_surface(
                mesh=mesh,
                count=2,
                face_weight=None,  # seed=random.randint(1, 1e8)
            )

            if any(point[2] < min_z for point in [point1, point2]):
                continue
            distance = pbu.get_distance(point1, point2)
            if (distance > max_width) or (distance < 1e-3):
                continue
            direction2 = point2 - point1
            if (
                abs(pbu.angle_between(target_vector, direction2) - np.pi / 2)
                > target_tolerance
            ):
                continue

            normal1 = extract_normal(mesh, index1)
            if normal1.dot(-direction2) < 0:
                normal1 *= -1
            error1 = pbu.angle_between(normal1, -direction2)

            normal2 = extract_normal(mesh, index2)
            if normal2.dot(direction2) < 0:
                normal2 *= -1
            error2 = pbu.angle_between(normal2, direction2)

            if (error1 > antipodal_tolerance) or (error2 > antipodal_tolerance):
                continue

            tool_from_grasp, _ = next(sample_grasp(obj, point1, point2))
            # score = combine_scores(
            #     score_overlap(intersector, point1, point2),
            #     score_torque(mesh, tool_from_grasp),
            # )

            world_T_obj = pbu.get_pose(obj, client=sim.client)
            world_T_parent = pbu.multiply(world_T_obj, pbu.invert(tool_from_grasp))

            if workspace_collision(sim, [world_T_parent], grasp=None, obstacles=[obj]):
                continue

            pbu.wait_if_gui(client=sim.client)
            closed_conf, _ = sim.get_group_limits(sim.gripper_group)
            closed_position = closed_conf[0] * (1 + 5e-2)
            return Grasp(
                attachment=Attachment(sim.robot, sim.tool_link, obj, tool_from_grasp),
                closed_position=closed_position,
            )
        return None

    return gen_fn
