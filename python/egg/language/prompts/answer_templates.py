from httpx import Response
from openai.types.chat.completion_create_params import ResponseFormat


DEFAULT_NULL_ANSWER_TEMPLATE = """
    [
        {{
            "answer": None
            "modality": {modality}
            "confidence": 0
            "explanation": "Empty graph, not enough information to answer the question."
        }} ]
"""

EVALUATOR_RESPONSE_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "evaluator",
        "schema": {
            "type": "object",
            "properties": {
                "accuracy": {
                    "type": "number",
                    "description": "How semantically similar is the generated answer to the ground truth answer on a scale of 0 to 1",
                },
                "explanation": {
                    "type": "string",
                    "description": "explanation for the accuracy evaluation.",
                },
            },
            "required": ["accuracy", "explanation"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

QUERY_RESPONSE_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_response",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the query. Note that the graph does not always contain enough information to answer the query. If the graph does not contain enough information, answer 'None'",
                },
                "modality": {
                    "type": "string",
                    "description": "The modality that the answer is returned in strictly based on the tag at the beginning of the query.",
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident you are on the answer, from 0-1, 0 being you have no clue how to answer, and 1 being absolutely confident in the answer. Furthermore, if the events that help you generate this answer is far away from the current time, decrease the confidence",
                },
                "explanation": {
                    "type": "string",
                    "description": "The explanation to the answer. Clearly state which object nodes, event nodes are involved with their node ID if you use them to generate the answer. Clearly state which edges are involved as well if they are used to generate the answer.",
                },
            },
            "required": ["answer", "modality", "confidence", "explanation"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PHASE_1_RESPONSE_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "phase_1",
        "schema": {
            "type": "object",
            "properties": {
                "start_year": {
                    "type": "number",
                    "description": "the start timestamp in year, return 0 if the query does not mention a time range",
                },
                "start_month": {
                    "type": "number",
                    "description": "the start timestamp in month, return 0 if the query does not mention a time range",
                },
                "start_day": {
                    "type": "number",
                    "description": "the start timestamp in day, return 0 if the query does not mention a time range",
                },
                "start_hour": {
                    "type": "number",
                    "description": "the start timestamp in hour, defaults to 0 if the query does not mention a time range",
                },
                "start_minute": {
                    "type": "number",
                    "description": "the start timestamp in minute, defaults to 0 if the query does not mention a time range",
                },
                "end_year": {
                    "type": "number",
                    "description": "the end timestamp in year, return the curent year if the query does not mention a time range",
                },
                "end_month": {
                    "type": "number",
                    "description": "the end timestamp in month, return the current month if the query does not mention a time range",
                },
                "end_day": {
                    "type": "number",
                    "description": "the end timestamp in day, return the current date if the query does not mention a time range",
                },
                "end_hour": {
                    "type": "number",
                    "description": "the end timestamp in hour, defaults to the current time if the query does not mention a time range",
                },
                "end_minute": {
                    "type": "number",
                    "description": "the end timestamp in minute, defaults to the current time if the query does not mention a time range",
                },
                "explanation_time": {
                    "type": "string",
                    "description": "explanation for the selection of the time range",
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "the array of strings of locations to look for the nodes, return all locations if the query does not mention any locations",
                },
                "explanation_locations": {
                    "type": "string",
                    "description": "explanation for the selection of the locations",
                },
            },
            "required": [
                "start_year",
                "start_month",
                "start_day",
                "start_hour",
                "start_minute",
                "end_year",
                "end_month",
                "end_day",
                "end_hour",
                "end_minute",
                "explanation_time",
                "locations",
                "explanation_locations",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
PHASE_2_RESPONSE_FORMAT: ResponseFormat = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_response",
        "schema": {
            "type": "object",
            "properties": {
                "object_nodes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "a list node ids of relevant object nodes to expand",
                },
                "explanation_objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "reasoning for choosing these object nodes",
                },
                "event_nodes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "a list node ids of relevant event nodes to expand",
                },
                "explanation_events": {
                    "type": "string",
                    "description": "reasoning for choosing these event nodes",
                },
            },
            "required": [
                "object_nodes",
                "explanation_objects",
                "event_nodes",
                "explanation_events",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
