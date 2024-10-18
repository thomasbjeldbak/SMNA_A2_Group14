#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install networkx python-louvain matplotlib


# In[91]:


import os
import pandas as pd
import networkx as nx
import logging

# Define the folder where the comment files are stored
comments_folder = 'youtube_data'

def load_comments_from_folder(comments_folder):
    """
    Load all YouTube comments from CSV files stored in the given folder.
    
    @param comments_folder: Path to the folder containing the CSV files
    @return: A DataFrame with all comments
    """
    all_comments = pd.DataFrame()
    
    for file_name in os.listdir(comments_folder):
        if file_name.startswith('youtube_comments_') and file_name.endswith('.csv'):
            file_path = os.path.join(comments_folder, file_name)
            try:
                # Try to read the CSV file
                video_comments = pd.read_csv(file_path)
                
                # If the CSV is empty, skip it
                if video_comments.empty:
                    logging.warning(f"Skipping empty file: {file_path}")
                    continue
                
                # Concatenate the comments into the master DataFrame
                all_comments = pd.concat([all_comments, video_comments], ignore_index=True)
            
            except pd.errors.EmptyDataError:
                # Handle empty files gracefully
                logging.warning(f"Skipping file with no data: {file_path}")
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
    
    return all_comments

def build_reply_graph(comments_df):
    """
    Build a reply graph from YouTube comments DataFrame.
    
    @param comments_df: DataFrame containing the scraped comments
    @return: A NetworkX graph with weighted edges based on replies
    """
    reply_graph = nx.DiGraph()  # Directed graph because replies are directed interactions
    
    for _, row in comments_df.iterrows():
        author = row['author']
        parent_id = row['parent_id']
        is_reply = row['is_reply']
        
        # Add nodes for the comment author and parent
        reply_graph.add_node(author)
        
        # If it's a reply, add an edge between the comment author and the parent author
        if is_reply:
            parent_author = comments_df.loc[comments_df['comment_id'] == parent_id, 'author'].values
            if len(parent_author) > 0:  # Check if parent author exists
                parent_author = parent_author[0]
                if author != parent_author:  # Avoid self-loops
                    # Add or update the edge weight (number of replies)
                    if reply_graph.has_edge(author, parent_author):
                        reply_graph[author][parent_author]['weight'] += 1
                    else:
                        reply_graph.add_edge(author, parent_author, weight=1)
    
    return reply_graph

def save_graph(graph, file_name="youtube_reply_graph"):
    """
    Save the reply graph to a file in GraphML and GEXF formats.
    
    @param graph: The NetworkX graph object
    @param file_name: The base name of the file (without extension)
    """
    graphml_path = f"{file_name}.graphml"
    gexf_path = f"{file_name}.gexf"
    
    # Save the graph in GraphML format
    nx.write_graphml(graph, graphml_path)
    logging.info(f"Graph saved to: {graphml_path}")
    
    # Save the graph in GEXF format (useful for Gephi)
    nx.write_gexf(graph, gexf_path)
    logging.info(f"Graph saved to: {gexf_path}")


# In[92]:


#Load all comments from the folder
comments_df = load_comments_from_folder(comments_folder)

#Check if there are any comments loaded
if not comments_df.empty:
    # Build the reply graph
    reply_graph = build_reply_graph(comments_df)

    # Save the reply graph to files in the current directory
    save_graph(reply_graph, file_name="youtube_reply_graph")
else:
    logging.error("No valid comments found in the folder.")


# In[93]:


nodes = list(reply_graph.nodes)
unique_nodes = set(nodes)

if len(nodes) == len(unique_nodes):
    print("No duplicate nodes found.")
else:
    print("Duplicate nodes detected!")


# In[94]:


import networkx as nx
import community.community_louvain as community_louvain
import random
import os

# Define the file paths (assuming they are saved in the current working directory)
youtube_graph_file = 'youtube_reply_graph.graphml'

# Check if the files exist in the current directory
if not os.path.exists(youtube_graph_file):
    raise FileNotFoundError(f"YouTube graph file '{youtube_graph_file}' not found.")

# Load the saved GraphML graphs
youtube_graph = nx.read_graphml(youtube_graph_file)

# Community Detection using Clique Percolation Method (CPM)
def clique_percolation_method(G, k):
    """
    Detect communities using Clique Percolation Method (CPM).
    This finds all cliques of size >= k and forms communities from overlapping cliques.
    
    @param G: NetworkX graph
    @param k: Minimum size of the clique
    @return: List of communities (sets of nodes)
    """
    G_undirected = G.to_undirected()  # Make sure the graph is undirected for clique finding
    cliques = [set(c) for c in nx.find_cliques(G_undirected) if len(c) >= k]  # Find cliques of size >= k
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(range(len(cliques)))  # Create a node for each clique
    
    # Add edges between cliques that share k-1 nodes
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if len(cliques[i] & cliques[j]) >= k - 1:
                clique_graph.add_edge(i, j)
    
    # Build communities from connected components in the clique graph
    communities = []
    for component in nx.connected_components(clique_graph):
        community = set()
        for idx in component:
            community.update(cliques[idx])
        communities.append(community)
    
    return communities

# Function to add centrality measures to the graph
def add_centrality_measures(graph):
    """
    Add degree, betweenness, and closeness centrality to each node in the graph.
    
    @param graph: The NetworkX graph
    """
    # Compute degree centrality
    degree_centrality = nx.degree_centrality(graph)
    
    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(graph)
    
    # Compute closeness centrality
    closeness_centrality = nx.closeness_centrality(graph)
    
    # Add centrality measures to each node
    for node in graph.nodes():
        graph.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
        graph.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
        graph.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)

# Apply CPM and Louvain on YouTube graph
def apply_community_detection(graph, graph_name):
    """
    Apply both CPM and Louvain community detection methods on the given graph.
    
    @param graph: NetworkX graph
    @param graph_name: Name of the graph (for output file naming)
    @return: A NetworkX graph with community attributes for both CPM and Louvain
    """
    # Apply CPM with k=3
    print(f"Applying CPM on {graph_name}...")
    cpm_communities = clique_percolation_method(graph, k=3)
    
    # Apply Louvain method with or without weights
    print(f"Applying Louvain on {graph_name}...")
    if 'replyNum' in nx.get_edge_attributes(graph, 'replyNum'):
        louvain_communities = community_louvain.best_partition(graph.to_undirected(), weight='replyNum')
    else:
        louvain_communities = community_louvain.best_partition(graph.to_undirected())
    
    # Add CPM community labels to the graph
    def add_communities_to_graph(graph, communities, attr_name):
        for comm_id, community in enumerate(communities):
            for node in community:
                if graph.has_node(node):
                    graph.nodes[node][attr_name] = comm_id
    
    # Add Louvain communities to the graph
    def add_louvain_communities_to_graph(graph, partition, attr_name):
        for node, comm_id in partition.items():
            if graph.has_node(node):
                graph.nodes[node][attr_name] = comm_id
    
    # Add both CPM and Louvain community attributes
    add_communities_to_graph(graph, cpm_communities, attr_name='CPM_community')
    add_louvain_communities_to_graph(graph, louvain_communities, attr_name='Louvain_community')
    
    # Add centrality measures to the graph
    print(f"Adding centrality measures to {graph_name}...")
    add_centrality_measures(graph)
    
    # Save the graph with community and centrality attributes
    output_file = f"{graph_name}_with_communities_and_centrality.graphml"
    nx.write_graphml(graph, output_file)
    print(f"Graph with CPM, Louvain, and centrality measures saved as {output_file}")


# In[95]:


# Apply community detection to YouTube graphs
apply_community_detection(youtube_graph, 'youtube_reply_graph')


# In[96]:


def extract_community_nodes(graph, community_attr):
    """
    Extract nodes for each community based on a given community attribute.
    
    @param graph: The NetworkX graph with community attributes.
    @param community_attr: The attribute name for the community (e.g., 'Louvain_community' or 'CPM_community').
    @return: A dictionary where the key is the community ID and the value is the list of nodes (users) in that community.
    """
    communities = {}
    
    for node, data in graph.nodes(data=True):
        community_id = data.get(community_attr)
        if community_id is not None:
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
    
    return communities



# In[97]:


# Extract Louvain communities for YouTube graphs
youtube_louvain_communities = extract_community_nodes(youtube_graph, community_attr='Louvain_community')

# Extract CPM communities for YouTube graphs
youtube_cpm_communities = extract_community_nodes(youtube_graph, community_attr='CPM_community')


# In[98]:


youtube_louvain_communities


# In[99]:


import pandas as pd
import os

# Define the folder where the comment files are stored
comments_folder = 'youtube_data'

def load_youtube_comments_from_folder(comments_folder):
    """
    Load all YouTube comments from CSV files stored in the given folder.
    
    @param comments_folder: Path to the folder containing the CSV files
    @return: A DataFrame with all comments
    """
    all_comments = pd.DataFrame()

    for file_name in os.listdir(comments_folder):
        if file_name.startswith('youtube_comments_') and file_name.endswith('.csv'):
            file_path = os.path.join(comments_folder, file_name)
            
            # Check if the file is empty (no columns to parse)
            if os.stat(file_path).st_size == 0:
                print(f"Skipping empty file: {file_name}")
                continue

            try:
                video_comments = pd.read_csv(file_path)
                if video_comments.empty:
                    print(f"No data in file: {file_name}")
                    continue  # Skip if the file has no rows
                
                all_comments = pd.concat([all_comments, video_comments], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f"Skipping file due to parsing error (EmptyDataError): {file_name}")
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
                continue  # Skip any other errors and continue

    return all_comments

def extract_youtube_comments_by_community(youtube_comments_df, communities):
    """
    Extract YouTube comments made by users in each community.
    
    @param youtube_comments_df: DataFrame containing YouTube comments data.
    @param communities: Dictionary where keys are community IDs and values are lists of users (nodes).
    @return: A dictionary where keys are community IDs and values are DataFrames with comments made by that community.
    """
    community_comments = {}

    # Filter comments made by the users in each community
    for community_id, users in communities.items():
        community_comments[community_id] = youtube_comments_df[youtube_comments_df['author'].isin(users)]
    
    return community_comments


# In[100]:


#Load all YouTube comments from the folder
youtube_comments_df = load_youtube_comments_from_folder(comments_folder)


# In[101]:


#Extract comments by Louvain community
youtube_comments_by_louvain_community = extract_youtube_comments_by_community(youtube_comments_df, youtube_louvain_communities)

#Extract comments by cpm community
youtube_comments_by_cpm_community = extract_youtube_comments_by_community(youtube_comments_df, youtube_louvain_communities)


# In[102]:


#Now you can explore the comments per community
for community_id, community_comments in youtube_comments_by_louvain_community.items():
    print(f"Community {community_id}:")
    print(community_comments.head())


# In[103]:


#Now you can explore the comments per community
for community_id, community_comments in youtube_comments_by_cpm_community.items():
    print(f"Community {community_id}:")
    print(community_comments.head())


# In[104]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_word_cloud_for_community(comments_by_community, community_name, platform, community_type):
    """
    Generate and plot a word cloud for a specific community based on the comments.

    @param comments_by_community: Dictionary containing DataFrames of comments by community.
    @param community_name: Name of the community (e.g., 'Reddit' or 'YouTube').
    @param platform: The platform ('reddit' or 'youtube') to distinguish between data structures.
    @param community_type: The type of community detection ('Louvain' or 'CPM').
    """
    for community_id, comments_df in comments_by_community.items():
        # Skip communities with no comments
        if comments_df.empty:
            print(f"Community {community_id} has no comments. Skipping.")
            continue

        # Check for the correct column based on the platform
        if platform == 'reddit':
            comment_column = 'comment_body'
        elif platform == 'youtube':
            comment_column = 'text'
        else:
            raise ValueError("Platform should be either 'reddit' or 'youtube'")

        # Join all comment text for the community
        all_text = ' '.join(comments_df[comment_column].astype(str))

        # Skip if the concatenated text is empty
        if not all_text.strip():
            print(f"Community {community_id} has no valid comment text. Skipping.")
            continue

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{community_name} - {community_type} Community {community_id}")
        plt.show()



# In[105]:


# Example usage for Youtube (Louvain and CPM communities):
generate_word_cloud_for_community(youtube_comments_by_louvain_community, "YouTube", platform='youtube', community_type='Louvain')


# In[107]:


# Example usage for Youtube (Louvain and CPM communities):
generate_word_cloud_for_community(youtube_comments_by_louvain_community, "YouTube", platform='youtube', community_type='CPM')


# In[130]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming the community_centrality_df is already loaded and contains community and centrality information

# Function to plot centrality distribution for a given community
def plot_centrality_distribution(df, centrality_column, community_id=None):
    """
    Plot the distribution of a specified centrality measure for a specific community.
    
    Parameters:
    - df: The dataframe containing community and centrality information.
    - centrality_column: The centrality measure to plot (e.g., 'degree_centrality').
    - community_id: The community ID to filter on (if None, plot for all communities).
    """
    if community_id is not None:
        df = df[df['community'] == community_id]
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[centrality_column], bins=30, color='blue', alpha=0.7)
    plt.title(f'Distribution of {centrality_column} in Community {community_id}')
    plt.xlabel(centrality_column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


#Select only numeric columns for centrality measures
numeric_columns = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']

# Calculate and display the average centrality for each community
average_centrality_by_community = community_centrality_df.groupby('community')[numeric_columns].mean()

# Display the average centrality values for each community
print(average_centrality_by_community)

#Find and display the top central nodes across all communities (by degree centrality)
top_central_nodes = community_centrality_df.sort_values(by='degree_centrality', ascending=False).head(10)
print("Top 10 nodes by degree centrality:")
print(top_central_nodes[['node', 'degree_centrality', 'community']])


# In[133]:


#Plot centrality distribution for a specific community (e.g., community 0)
plot_centrality_distribution(community_centrality_df, 'degree_centrality', community_id=198)


# In[123]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@Crash1025Bandicoot7']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[122]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@DaleRV']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[125]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@johnwilliams8397']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[126]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@diaperdon-g5b']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[128]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@MSNBCult']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[129]:


# Set pandas option to display the full text in cells without truncation
pd.set_option('display.max_colwidth', None)

# Filter the DataFrame to get all comments made by user
comments_df = youtube_comments_df[youtube_comments_df['author'] == '@JesusSaysMAGAtrashMAGAfilth']

# Display the full text of the first 5 comments
print(comments_df['text'].head(5))


# In[ ]:




