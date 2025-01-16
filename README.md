# Tiny Task and Motion Planning

A minimal task and motion planning package for quick prototyping

## Install

Create a virtual environment:

`conda create --name tiny_tamp python=3.8`

Build:

`python -m pip install -e .`

Install optional dependencies via:

`pip install '.[dev]'`

## Use

`python tamp.py --vis-sim`

# Pre-commit

`pre-commit run --all-files`

## Debugging

To debug, you can visualize the belief in the digital twin simulator with
`python tamp.py --vis-belief`

By adding the following, you can pause the planning and view the current state of the simulator in pybullet.
Pressing enter in the terminal will resume planning.
`pbu.wait_if_gui(client=s...)`


# Running on real robot

In a separate terminal window in your ros python environment, run `panda_server.py`. If things boot up correctly, you should see "Waiting for request...".

Running the planner will communicate with this server via zmq.

`python tamp.py --vis-sim --real-robot`

## TODO

- [ ] create statistics upon failure
- [x] test the antipodal grasp sampler
- [ ] ycb objects
- [ ] test on real robot
- [x] replace fastik
- [ ] replace motion planner with curobo (or something better)
