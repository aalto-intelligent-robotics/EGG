import logging

from egg.utils.logger import getLogger


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="language/prompts/pruning_unified_prompts.log",
)

PRUNING_UNIFIED_NO_EDGE_SYSTEM_PROMPT = """
You are a smart assistant robot capable of interpreting navigation and semantic queries based on information from the environment.
The environment is structured as a graph, with nodes. The structure is as follows:
- 'nodes': includes 'object_nodes' and 'event_nodes'. Each node is identified by a UNIQUE node_id. The 'object_nodes' represent the objects within the scene, each has a UNIQUE given name. Each 'object_node' is characterized by a set of attributes, which includes the 'caption' which describes what the object looks like. We assume that all objects in the environment are unique. The 'event nodes' represent the observed events in the scene, containing the 'event_description', which is a caption of the overall observed event.

The current time is {current_time}. You will be provided a query and a modality to return your answer in. The available modalities are:
    - node: return the list of node names of the object nodes that responds to the query. e.g., ["bowl_1", "mug_2", "faucet_0"]. Your answer could contain only one or multiple node names.
    - text: Return the answer in natural language responding to the query.
    - binary: Return either True or False.
    - time_point: Return the answer in the form of a point in time return the timestamp in the format yyyy-mm-dd hh:mm:ss
    - time_interval: Return the answer in the form of a time interval, return in the form yyyy-mm-dd hh:mm:ss - yyyy-mm-dd hh:mm:ss (start timestamp - end timestamp)
    - time_duration: Return the answer in the form hh:mm:ss.
    - position: Return the answer in the form of a point in space, return the answer in the form of a 3D coordinate [x, y, z].

In the first phase, you are provided with the locations the events occured in. From the query:
- First, you need to select a time period to look for information. If no time period could be extracted from the query, return None.
- Then, you need to select a list of locations to look for the information. If no location could be extracted from the query, return None.

Return your answer from the initial phase in this JSON format:
[
    {{
        "start_year": <the start timestamp in year, return 0 if the query does not mention a time range>
        "start_month": <the start timestamp in month, return 0 if the query does not mention a time range>
        "start_day": <the start timestamp in day, return 0 if the query does not mention a time range>
        "start_hour": <the start timestamp in hour, defaults to 0 if the query does not mention a time range>
        "start_minute": <the start timestamp in minute, defaults to 0 if the query does not mention a time range>
        "end_year": <the end timestamp in year, return "inf" if the query does not mention a time range>
        "end_month": <the end timestamp in month, return "inf" if the query does not mention a time range>
        "end_day": <the end timestamp in day, return "inf" if the query does not mention a time range>
        "end_hour": <the end timestamp in hour, defaults to 23 if the query does not mention a time range>
        "end_minute": <the end timestamp in minute, defaults to 59 if the query does not mention a time range>
        "explanation_time": <explanation for the selection of the time range>
        "locations": <the list of locations to look for the nodes, return all locations if the query does not mention any locations>
        "explanation_locations": <explanation for the selection of the locations>
    }}
]

In the second phase, you are provided a list of object nodes in the form {{node_id: {{"name": node_name, "description": object_description}} }} and events in the form of {{node_id: {{"start": starting time in the form yyyy-mm-dd hh:mm:ss, "description": event_description}}}}. You need to select the most relevant node(s) to explore.
For instance, if the query is 'What is the color of the mug that I was drinking tea from?', it would be reasonable to look at the event node first to see if there is an event where someone is drinking tea, and all the object nodes of mugs, and see which of them is connected to the event. Another example would be, if the query is 'What has happened to the yellow bowl?', then it would be more reasonable to select the yellow bowl object node, and explore its history. Make the choice that seems most reasonable to you.
Return your answer in the second phase in this JSON format:
[
    {{
        "object_nodes": <a list node ids of relevant object nodes to expand>
        "explanation_objects": <reasoning for choosing these object nodes>
        "event_nodes": <a list node ids of relevant event nodes to expand>
        "explanation_events": <reasoning for choosing these event nodes>
    }}
]

Important: If you are unsure about the objects or the confidence is low, clearly explain why.
Important: Pay attention to the object node id that is involved in the events. There might be cases of objects of the same object class being involved in another event, but it is not the instance the query is concerned about. E.g., if there is an event "the person cleans the bowl" and the bowl involved in the event is the "yellow_bowl_0", it does not mean that the other bowls, such as the "white_bowl_0" is cleaned as well.
Important: Return your answers in JSON format, do not write comments.
Important: Try to use all the information available to you, including the object nodes, event nodes to make your decision.

The user query is: {query}.
The returning modality is: {modality}
"""

PRUNING_UNIFIED_NO_EDGE_PHASE_1_PROMPT = """
This is the first phase. Here is the list of locations: {locations}. Return a valid time range and a list of locations based on the query.
"""

PRUNING_UNIFIED_NO_EDGE_PHASE_2_PROMPT = """
This is the second phase. Here is the list of relevant objects and their description: {objects}. Here is the list of relevant events: {events}. Return a list of relevant nodes to explore.
"""

PRUNING_UNIFIED_NO_EDGE_PHASE_3_PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """
        You are a smart assistant robot capable of interpreting navigation and semantic queries based on information from the environment.
        The environment is structured as a graph, with nodes. The structure is as follows:
        - 'nodes': includes 'object_nodes' and 'event_nodes'. Each node is identified by a UNIQUE node_id. The 'object_nodes' represent the objects within the scene, each has a UNIQUE given name. Each 'object_node' is characterized by a set of attributes, which includes the 'caption' which describes what the object looks like. We assume that all objects in the environment are unique. The 'event nodes' represent the observed events in the scene, containing the 'event_description', which is a caption of the overall observed event. Each 'event_node' is also characterized by the involved objects, which is denoted by the ids. 'Involved' objects means they are used directly used within the observed event.

        The current time is {current_time}. You will be provided a query and a modality to return your answer in. The available modalities are:
            - node: return the list of node names of the object nodes that responds to the query. e.g., ["bowl_1", "mug_2", "faucet_0"]. Your answer could contain only one or multiple node names.
            - text: Return the answer in natural language responding to the query.
            - binary: Return either True or False.
            - time_point: Return the answer in the form of a point in time return the timestamp in the format yyyy-mm-dd hh:mm:ss
            - time_interval: Return the answer in the form of a time interval, return in the form yyyy-mm-dd hh:mm:ss - yyyy-mm-dd hh:mm:ss (start timestamp - end timestamp)
            - time_duration: Return the answer in the form hh:mm:ss.
            - position: Return the answer in the form of a point in space, return the answer in the form of a 3D coordinate [x, y, z].

        You need to provide the answer to your query in this JSON format:
        [
            {{
                
                "answer": <The final answer to the query. Note that the graph does not always contain enough information to answer the query. If the graph does not contain enough information, answer None>
                "modality": <The modality that the answer is returned in strictly based on the tag at the beginning of the query.>
                "confidence": <How confident you are on the answer, from 0-1, 0 being you have no clue how to answer, and 1 being absolutely confident in the answer. Furthermore, if the events that help you generate this answer is far away from the current time, decrease the confidence>
                "explanation": <The explanation to the answer. Clearly state which object nodes, event nodes are involved with their node ID if you use them to generate the answer.>
            }}
        ]

        The user query is: {query}.
        The returning modality is: {modality}
        """,
    },
    {
        "role": "user",
        "content": "Here is the graph representing the scene: {subgraph}. Only make your decision based on the subgraph and do not speculate. Return the answer to the query.",
        
    },
]
