MICROSERVICE_DAGS = {
    "Data_Heavy_DAG": {
        "probability": 0.1,
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
        "probability": 0.05,
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
        "probability": 0.10,
        "nodes": {
            "MS_41612": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
            "MS_10041": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
            "MS_27421": {"image_mb": 50, "state_mb": 10, "is_stateful": True},
        },
        "edges": {
            ("MS_41612", "MS_10041"): 1,
            ("MS_10041", "MS_27421"): 2,
        }
    },
    "FanIn_Aggregator_1": {
        "probability": 0.05,
        "nodes": {
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_10105": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_100": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_10110": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_1010": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_10105", "MS_37691"): 18,
            ("MS_100", "MS_37691"): 1,
            ("MS_10110", "MS_37691"): 40,
            ("MS_1010", "MS_37691"): 28,
        }
    },
    "FanIn_Aggregator_2": {
        "probability": 0.05,
        "nodes": {
            "MS_10041": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
            "MS_10097": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_10019": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_10064": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_10041", "MS_27421"): 3,
            ("MS_10097", "MS_27421"): 4,
            ("MS_10019", "MS_27421"): 1,
            ("MS_10064", "MS_27421"): 5,
        }
    },
    "FanIn_Aggregator_3": {
        "probability": 0.05,
        "nodes": {
            "MS_1052": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_1042": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_10810": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_10673": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_46825": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_1052", "MS_46825"): 2,
            ("MS_1042", "MS_1052"): 2,
            ("MS_1042", "MS_46825"): 10,
            ("MS_10810", "MS_46825"): 28,
            ("MS_10673", "MS_46825"): 1,
        }
    },
    "FanOut_Broadcaster_1": {
        "probability": 0.10,
        "nodes": {
            "MS_20664": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_9570": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_20896": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_19709": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_1303": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_9570", "MS_1303"): 5029,
            ("MS_9570", "MS_19709"): 4,
            ("MS_9570", "MS_20664"): 24106,
            ("MS_9570", "MS_20896"): 6,
            ("MS_19709", "MS_20896"): 554,
        }
    },
    "FanOut_Broadcaster_2": {
        "probability": 0.10,
        "nodes": {
            "MS_11139": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_12729": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "UNAVAILABLE": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_12086": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_1415": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("UNAVAILABLE", "MS_11139"): 348,
            ("UNAVAILABLE", "MS_12086"): 91,
            ("UNAVAILABLE", "MS_12729"): 106,
            ("UNAVAILABLE", "MS_1415"): 79,
        }
    },
    "FanOut_Broadcaster_3": {
        "probability": 0.10,
        "nodes": {
            "MS_13448": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
            "MS_49817": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_40912": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_58269": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_49817", "MS_58269"): 84,
            ("MS_40912", "MS_13448"): 116,
            ("MS_40912", "MS_49817"): 30,
            ("MS_40912", "MS_58269"): 21335,
        }
    },
    "Diamond_DAG_1": {
        "probability": 0.10,
        "nodes": {
            "MS_37235": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_15934": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "USER": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "UNKNOWN": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
        },
        "edges": {
            ("MS_37235", "MS_15934"): 805,
            ("MS_37235", "UNKNOWN"): 3509,
            ("MS_15934", "MS_37235"): 1168,
            ("MS_15934", "UNKNOWN"): 60146,
            ("USER", "MS_15934"): 5456,
            ("USER", "MS_37235"): 700,
            ("USER", "UNKNOWN"): 9128,
            ("UNKNOWN", "MS_15934"): 2341,
            ("UNKNOWN", "MS_37235"): 152,
        }
    },
    "Diamond_DAG_2": {
        "probability": 0.10,
        "nodes": {
            "MS_15284": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_7401": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_28467": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_37027": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_15284", "MS_28467"): 19644,
            ("MS_15284", "MS_7401"): 17,
            ("MS_7401", "MS_37027"): 2,
            ("MS_37027", "MS_28467"): 86,
        }
    },
    "Diamond_DAG_3": {
        "probability": 0.10,
        "nodes": {
            "MS_51052": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_55085": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_4660": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_7766": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_51052", "MS_55085"): 16808,
            ("MS_4660", "MS_51052"): 2,
            ("MS_4660", "MS_55085"): 31764,
            ("MS_4660", "MS_7766"): 459,
            ("MS_7766", "MS_55085"): 6,
        }
    }
}
