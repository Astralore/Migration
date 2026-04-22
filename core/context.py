"""
SLA-based migration trigger for edge microservice migration.
Supports both reactive and proactive (trajectory-prediction) modes.
"""

from core.geo import haversine_distance

DISTANCE_THRESHOLD_KM = 15.0
SLA_REWARD_THRESHOLD = -5.0


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

    Parameters
    ----------
    predicted_locations : list of (lat, lon) or None
        Future H-step predicted positions.  When provided the function also
        fires if *any* future position would breach the distance threshold
        against the current gateway server.

    Returns
    -------
    bool — True if migration should be attempted
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
            if future_dist > DISTANCE_THRESHOLD_KM:
                return True

    return False
