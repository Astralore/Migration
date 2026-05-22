MICROSERVICE_DAGS = {
    "FanIn_Aggregator_1": {
        "probability": 0.1411,
        "nodes": {
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_53154": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_63670": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_15284": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_65114": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_66431": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_52363": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_63670", "MS_37691"): 25609,
            ("MS_65114", "MS_37691"): 19685,
            ("MS_52363", "MS_37691"): 7562,
            ("MS_66431", "MS_37691"): 8387,
            ("MS_53154", "MS_37691"): 25818,
            ("MS_15284", "MS_37691"): 22403,
        }
    },
    "FanIn_Aggregator_2": {
        "probability": 0.0213,
        "nodes": {
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
            "MS_49617": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_22179": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_28081": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_30441": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_53473": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_65240": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_22179", "MS_27421"): 3358,
            ("MS_49617", "MS_27421"): 3782,
            ("MS_65240", "MS_27421"): 1955,
            ("MS_53473", "MS_27421"): 2308,
            ("MS_28081", "MS_27421"): 2768,
            ("MS_30441", "MS_27421"): 2356,
        }
    },
    "FanIn_Aggregator_3": {
        "probability": 0.0554,
        "nodes": {
            "MS_46825": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_4660": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_51052": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_48385": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_155": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_25557": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_32977": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_48385", "MS_46825"): 4156,
            ("MS_25557", "MS_46825"): 2915,
            ("MS_32977", "MS_46825"): 2893,
            ("MS_4660", "MS_46825"): 23240,
            ("MS_4660", "MS_51052"): 2,
            ("MS_155", "MS_46825"): 3658,
            ("MS_51052", "MS_46825"): 6077,
        }
    },
    "FanOut_Broadcaster_1": {
        "probability": 0.1040,
        "nodes": {
            "MS_51052": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_24560": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_55085": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_32139": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_20973": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_46825": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_66332": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_51052", "MS_20973"): 10185,
            ("MS_51052", "MS_24560"): 28596,
            ("MS_51052", "MS_32139"): 14194,
            ("MS_51052", "MS_46825"): 6077,
            ("MS_51052", "MS_55085"): 16808,
            ("MS_51052", "MS_66332"): 4829,
        }
    },
    "FanOut_Broadcaster_2": {
        "probability": 0.0832,
        "nodes": {
            "MS_4660": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_55085": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_46825": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_14728": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_8234": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_2827": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_4660", "MS_14728"): 1604,
            ("MS_4660", "MS_2827"): 1232,
            ("MS_4660", "MS_37691"): 5381,
            ("MS_4660", "MS_46825"): 23240,
            ("MS_4660", "MS_55085"): 31764,
            ("MS_4660", "MS_8234"): 1268,
        }
    },
    "FanOut_Broadcaster_3": {
        "probability": 0.0674,
        "nodes": {
            "MS_53154": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_28467": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_28245": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_67767": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_1076": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_73347": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_1076", "MS_37691"): 508,
            ("MS_28245", "MS_37691"): 2,
            ("MS_53154", "MS_1076"): 32,
            ("MS_53154", "MS_28245"): 115,
            ("MS_53154", "MS_28467"): 25724,
            ("MS_53154", "MS_37691"): 25818,
            ("MS_53154", "MS_67767"): 73,
            ("MS_53154", "MS_73347"): 2,
        }
    },
    "Diamond_DAG_1": {
        "probability": 0.0685,
        "nodes": {
            "MS_9570": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_20664": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_66431": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_7103": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_66431", "MS_37691"): 8387,
            ("MS_9570", "MS_20664"): 24106,
            ("MS_7103", "MS_37691"): 4751,
            ("MS_20664", "MS_37691"): 230,
            ("MS_20664", "MS_66431"): 10285,
            ("MS_20664", "MS_7103"): 5374,
        }
    },
    "Diamond_DAG_2": {
        "probability": 0.0316,
        "nodes": {
            "MS_9570": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_20664": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_53049": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
        },
        "edges": {
            ("MS_53049", "MS_27421"): 415,
            ("MS_9570", "MS_20664"): 24106,
            ("MS_9570", "MS_53049"): 14,
            ("MS_20664", "MS_27421"): 1,
        }
    },
    "Diamond_DAG_3": {
        "probability": 0.0312,
        "nodes": {
            "MS_9570": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_20664": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_37831": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
        },
        "edges": {
            ("MS_37831", "MS_27421"): 56,
            ("MS_9570", "MS_20664"): 24106,
            ("MS_9570", "MS_37831"): 10,
            ("MS_20664", "MS_27421"): 1,
        }
    },
    "Pipeline_Chain_1": {
        "probability": 0.0173,
        "nodes": {
            "MS_48385": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_58845": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_71712": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
        },
        "edges": {
            ("MS_71712", "MS_27421"): 1156,
            ("MS_58845", "MS_71712"): 1031,
            ("MS_48385", "MS_58845"): 11260,
        }
    },
    "Pipeline_Chain_2": {
        "probability": 0.0192,
        "nodes": {
            "MS_27283": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_58845": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_71712": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
        },
        "edges": {
            ("MS_27283", "MS_27421"): 703,
            ("MS_27283", "MS_58845"): 11965,
            ("MS_58845", "MS_71712"): 1031,
            ("MS_71712", "MS_27421"): 1156,
        }
    },
    "Pipeline_Chain_3": {
        "probability": 0.0555,
        "nodes": {
            "MS_9570": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_20664": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_66431": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
        },
        "edges": {
            ("MS_66431", "MS_37691"): 8387,
            ("MS_9570", "MS_20664"): 24106,
            ("MS_20664", "MS_37691"): 230,
            ("MS_20664", "MS_66431"): 10285,
        }
    },
    "Data_Heavy_DAG_1": {
        "probability": 0.1206,
        "nodes": {
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_53154": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_63670": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_15284": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_65114": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_63670", "MS_37691"): 25609,
            ("MS_65114", "MS_37691"): 19685,
            ("MS_53154", "MS_37691"): 25818,
            ("MS_15284", "MS_37691"): 22403,
        }
    },
    "Data_Heavy_DAG_2": {
        "probability": 0.0158,
        "nodes": {
            "MS_27421": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
            "MS_49617": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_22179": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_28081": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_30441": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_22179", "MS_27421"): 3358,
            ("MS_49617", "MS_27421"): 3782,
            ("MS_28081", "MS_27421"): 2768,
            ("MS_30441", "MS_27421"): 2356,
        }
    },
    "Data_Heavy_DAG_3": {
        "probability": 0.0479,
        "nodes": {
            "MS_46825": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_4660": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_51052": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_48385": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_155": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_48385", "MS_46825"): 4156,
            ("MS_4660", "MS_46825"): 23240,
            ("MS_4660", "MS_51052"): 2,
            ("MS_155", "MS_46825"): 3658,
            ("MS_51052", "MS_46825"): 6077,
        }
    },
    "Compute_Heavy_DAG_1": {
        "probability": 0.0644,
        "nodes": {
            "MS_9570": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_20664": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_66431": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_66701": {"image_mb": 150, "state_mb": 512, "is_stateful": True},
            "MS_7103": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_9570", "MS_20664"): 24106,
            ("MS_20664", "MS_66431"): 10285,
            ("MS_20664", "MS_66701"): 10199,
            ("MS_20664", "MS_7103"): 5374,
        }
    },
    "Compute_Heavy_DAG_2": {
        "probability": 0.0301,
        "nodes": {
            "MS_15284": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_7401": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_30273": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_23945": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_7401", "MS_37691"): 235,
            ("MS_30273", "MS_37691"): 16,
            ("MS_15284", "MS_23945"): 32,
            ("MS_15284", "MS_37691"): 22403,
            ("MS_15284", "MS_7401"): 17,
        }
    },
    "Compute_Heavy_DAG_3": {
        "probability": 0.0255,
        "nodes": {
            "MS_65114": {"image_mb": 200, "state_mb": 0, "is_stateful": False},
            "MS_37691": {"image_mb": 150, "state_mb": 128, "is_stateful": True},
            "MS_8445": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_53171": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
            "MS_70714": {"image_mb": 50, "state_mb": 0, "is_stateful": False},
        },
        "edges": {
            ("MS_65114", "MS_37691"): 19685,
            ("MS_53171", "MS_37691"): 31,
            ("MS_53171", "MS_65114"): 22,
            ("MS_8445", "MS_65114"): 26,
            ("MS_70714", "MS_65114"): 16,
        }
    },
}