class FSMState:
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    DRIVE_OUT = "DRIVE_OUT"
    REACTIVE_AVOID = "REACTIVE_AVOID"
    AVOID_PLANNING = "AVOID_PLANNING"
    PATH_FOLLOWING = "PATH_FOLLOWING"
    UTURN_PLANNING = "UTURN_PLANNING"
    RECOVERY = "RECOVERY"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class FSMEvent:
    LaneDetected = "LaneDetected"
    LaneLost = "LaneLost"
    ObstacleDetected = "ObstacleDetected"
    ObstacleCleared = "ObstacleCleared"
    PlannerSuccess = "PlannerSuccess"
    PlannerFailed = "PlannerFailed"
    DriveOutRequested = "DriveOutRequested"
    DriveOutCompleted = "DriveOutCompleted"
    UTurnRequested = "UTurnRequested"
    UTurnFinished = "UTurnFinished"
    PathCompleted = "PathCompleted"
    LocalizationLost = "LocalizationLost"
    RecoverySuccess = "RecoverySuccess"
    RecoveryFailed = "RecoveryFailed"
    Emergency = "Emergency"


class FSMCoordinator:
    """
    Finite State Machine Coordinator. Decouples state transition logic from ROS2 and execution.
    """
    def __init__(self, initial_state=FSMState.TRACKING):
        self.state = initial_state
        self.state_history = [initial_state]
        self.state_before_planning = None

    def get_state(self):
        return self.state

    def set_state(self, state):
        if self.state != state:
            self.state_before_planning = self.state
            self.state = state
            self.state_history.append(state)
            return True
        return False

    def handle_event(self, event):
        """
        Calculates and performs transition based on the incoming event.
        Returns the new state if transitioned, or None.
        """
        old_state = self.state
        new_state = old_state

        if old_state == FSMState.IDLE:
            if event == FSMEvent.LaneDetected:
                new_state = FSMState.TRACKING
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.TRACKING:
            if event == FSMEvent.ObstacleDetected:
                new_state = FSMState.REACTIVE_AVOID
            elif event == FSMEvent.DriveOutRequested:
                new_state = FSMState.DRIVE_OUT
            elif event == FSMEvent.UTurnRequested:
                new_state = FSMState.UTURN_PLANNING
            elif event == FSMEvent.LaneLost:
                new_state = FSMState.RECOVERY
            elif event == FSMEvent.LocalizationLost:
                new_state = FSMState.RECOVERY
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.DRIVE_OUT:
            if event == FSMEvent.UTurnRequested:
                new_state = FSMState.UTURN_PLANNING
            elif event == FSMEvent.DriveOutCompleted:
                new_state = FSMState.IDLE
            elif event == FSMEvent.LaneDetected:
                new_state = FSMState.TRACKING
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.REACTIVE_AVOID:
            if event == FSMEvent.ObstacleCleared:
                new_state = FSMState.TRACKING
            elif event == FSMEvent.PlannerFailed:
                new_state = FSMState.AVOID_PLANNING
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.AVOID_PLANNING:
            if event == FSMEvent.PlannerSuccess:
                new_state = FSMState.PATH_FOLLOWING
            elif event == FSMEvent.PlannerFailed:
                new_state = FSMState.RECOVERY
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.PATH_FOLLOWING:
            if event == FSMEvent.PathCompleted:
                new_state = FSMState.TRACKING
            elif event == FSMEvent.ObstacleDetected:
                new_state = FSMState.REACTIVE_AVOID
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.UTURN_PLANNING:
            if event == FSMEvent.PlannerSuccess:
                new_state = FSMState.PATH_FOLLOWING
            elif event == FSMEvent.PlannerFailed:
                new_state = FSMState.RECOVERY
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP


        elif old_state == FSMState.RECOVERY:
            if event == FSMEvent.RecoverySuccess:
                new_state = FSMState.TRACKING
            elif event == FSMEvent.RecoveryFailed:
                new_state = FSMState.EMERGENCY_STOP
            elif event == FSMEvent.Emergency:
                new_state = FSMState.EMERGENCY_STOP

        elif old_state == FSMState.EMERGENCY_STOP:
            # Sticky state, can only be reset by external intervention (re-initialization)
            pass

        if new_state != old_state:
            self.set_state(new_state)
            return new_state
        return None
