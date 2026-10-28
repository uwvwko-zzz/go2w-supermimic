Go2W model copied from the user-provided
`unitree_rl_mjlab/src/assets/robots/unitree_go2w/xmls` directory.
Only meshes referenced by `go2w.xml` are included. The XML is preserved;
`robot/config.py` names unnamed collision geoms, clears collision margins for Warp
MULTICCD compatibility, delegates solver options to the task, and adds the twelve leg actuators
through mjlab. The four XML velocity actuators retain their original gains,
speed limits and torque limits. These parameters are not hardware-validated.

The Go2-derived meshes are accompanied by the existing
`../LICENSE_menagerie_go2` attribution in this repository.
