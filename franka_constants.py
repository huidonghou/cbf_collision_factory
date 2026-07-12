class Franka_constants:
    # Defining constants, and the order is 
    # 0: base pan(rotate left/right, parallel to table)
    # 1: Shoulder pitch: tilt the upper arm forward and backward
    # 2: shoulder roll/bicep twist: rotate the upper arm around its axis
    # 3: elbow pitch: bends and straightens the elbow
    # 4: forearm twist: twists lower arm along the axis
    # 5: wrist pitch: tilts head up and down
    # 6: wrist roll: twist hand/gripper like screwdriver

    ARM_DOF = 7
    FRANKA_Q_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    FRANKA_Q_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
    FRANKA_DQ_LIM  = [ 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100]

    # This is the absolute maximum continuous torque (rotational force) the motors can output, measured in Newton-meters (N-m).
    # Two additional indices here: left finder that slide linearly left/right to grip
    # and the right finger that slides linearly to meet the left finger
    # So there's a caveat here: mix of units, first 7 is torque(N per meter), the last two is directly force (Newton)
    TAU_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 5.0, 5.0]