"""Velocity ranges scheduled in learner iterations, independent of sim substeps."""
STAGES = (
  (0, (-.3,.8), (-.2,.2), (-.8,.8)),
  (1000, (-.5,1.), (-.4,.4), (-1.,1.)),
  (2500, (-.8,1.5), (-.6,.6), (-1.5,1.5)),
  (5000, (-1.5,1.8), (-.8,.8), (-2.,2.)),
  (7500, (-2.,2.), (-1.,1.), (-3.,3.)),
)


def set_command_stage(cfg, iteration, stage=None):
  index = stage if stage is not None else max(i for i, item in enumerate(STAGES) if max(0,iteration) >= item[0])
  _, x, y, yaw = STAGES[index]
  cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z = x, y, yaw
  return index
