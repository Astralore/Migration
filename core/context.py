"""
SLA-based migration trigger for edge microservice migration.
Supports both reactive and proactive (trajectory-prediction) modes.
Returns explicit trigger_type for asymmetric migration cost calculation.
"""

from core.geo import haversine_distance

# Reactive threshold: actual SLA violation (user perceives outage)
DISTANCE_THRESHOLD_KM = 15.0
SLA_REWARD_THRESHOLD = -5.0

# Proactive warning threshold: early-warning buffer zone
# Triggers preemptive migration BEFORE actual violation occurs
PROACTIVE_WARNING_KM = 13.0

# Trigger type constants
TRIGGER_REACTIVE = 'REACTIVE'
TRIGGER_PROACTIVE = 'PROACTIVE'


def check_sla_violation(user_lat, user_lon,
                        gateway_server_lat, gateway_server_lon,
                        current_dag_reward):
    """
    Reactive trigger: current spatial or QoS violation.

    Returns True when EITHER condition is met:
      1. Spatial  — user-to-gateway distance > 15 km
      2. QoS     — current DAG reward < SLA_REWARD_THRESHOLD
    """
    dist = haversine_distance(user_lat, user_lon,
                              gateway_server_lat, gateway_server_lon)
    spatial_violation = dist > DISTANCE_THRESHOLD_KM
    qos_violation = current_dag_reward < SLA_REWARD_THRESHOLD
    return spatial_violation or qos_violation


def check_proactive_sla_violation(user_lat, user_lon,
                                  gateway_server_lat, gateway_server_lon,
                                  current_dag_reward,
                                  predicted_locations=None):
    """
    Proactive trigger: reactive check PLUS look-ahead over predicted trajectory.
    Returns True if migration should be attempted.
    """
    if check_sla_violation(user_lat, user_lon,
                           gateway_server_lat, gateway_server_lon,
                           current_dag_reward):
        return True

    if predicted_locations:
        for pred_lat, pred_lon in predicted_locations:
            future_dist = haversine_distance(pred_lat, pred_lon,
                                             gateway_server_lat,
                                             gateway_server_lon)
            # Use warning threshold for proactive trigger (earlier than reactive)
            if future_dist > PROACTIVE_WARNING_KM:
                return True

    return False


def get_trigger_type(user_lat, user_lon,
                     gateway_server_lat, gateway_server_lon,
                     current_dag_reward,
                     predicted_locations=None,
                     proactive_enabled=False):
    """
    Determine the trigger type for migration decision.

    Returns
    -------
    str or None
        'REACTIVE'  — current state already violates SLA (user perceives outage)
        'PROACTIVE' — predicted future violation (preemptive, user unaware)
        None        — no trigger needed
    """
    reactive_violation = check_sla_violation(
        user_lat, user_lon,
        gateway_server_lat, gateway_server_lon,
        current_dag_reward,
    )

    if reactive_violation:
        return TRIGGER_REACTIVE

    if proactive_enabled and predicted_locations:
        for pred_lat, pred_lon in predicted_locations:
            future_dist = haversine_distance(
                pred_lat, pred_lon,
                gateway_server_lat, gateway_server_lon,
            )
            # Use warning threshold for proactive trigger (13 km < 15 km)
            # This creates a 2km buffer zone for preemptive migration
            if future_dist > PROACTIVE_WARNING_KM:
                return TRIGGER_PROACTIVE

    return None
