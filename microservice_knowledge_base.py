MICROSERVICE_DAGS = {
    "Data_Heavy_DAG": {
        "probability": 0.40,
        "nodes": {
            "MS_37295": {"image_mb": 150, "state_mb": 0, "is_stateful": False},
            "MS_4281": {"image_mb": 150, "state_mb": 0, "is_stateful": False},
            "MS_8099": {"image_mb": 150, "state_mb": 0, "is_stateful": False},
            "MS_37374": {"image_mb": 150, "state_mb": 256, "is_stateful": True},
            "UNKNOWN": {"image_mb": 150, "state_mb": 0, "is_stateful": False},
            "USER": {"image_mb": 150, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_4281", "MS_8099"): 14,
            ("MS_8099", "MS_37295"): 8,
            ("MS_8099", "MS_37374"): 13937,
            ("UNKNOWN", "MS_4281"): 7,
            ("UNKNOWN", "MS_8099"): 4587,
            ("USER", "MS_8099"): 12,
            ("USER", "UNKNOWN"): 617,
        }
    },
    "Compute_Heavy_DAG": {
        "probability": 0.35,
        "nodes": {
            "UNKNOWN": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_10097": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_10370": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_50265": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("UNKNOWN", "MS_10370"): 3,
            ("UNKNOWN", "MS_50265"): 22707,
            ("MS_10097", "UNKNOWN"): 2,
            # ("MS_10370", "UNKNOWN"): 3,
        }
    },
    "IoT_Lightweight_DAG": {
        "probability": 0.25,
        "nodes": {
            "MS_41612": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
            "MS_10041": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
            "MS_27421": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
        },
        "edges": {
            ("MS_41612", "MS_10041"): 1,
            ("MS_10041", "MS_27421"): 2,
        }
    }
}