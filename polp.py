graph = {
    '1': ['2', '7', '8'],
    '2': ['3', '6'],
    '3': ['4', '5'],
    '4': [],
    '5': [],
    '6': [],
    '7': [],
    '8': ['9', '12'],
    '9': ['10', '11'],
    '10': [],
    '11': [],
    '12': []
}

def dfs(graph, node, goal, visited):
    if node not in visited:
        visited.append(node)
        if node == goal:
            return True
        for n in graph[node]:
            if dfs(graph, n, goal, visited):
                return True
    return False

visited = []
goal = '6'
dfs(graph, '1', goal, visited)
print(visited)










from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer,TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Download required resources
#nltk.download()
nltk.download('punkt')
nltk.download('wordnet')

# Input text
text = input("Enter Text:\n")

# Sentence Tokenization
sentences = sent_tokenize(text)
print("\nSentence Tokenization:")
print(sentences)

# Word Tokenization
words = word_tokenize(text)
print("\nWord Tokenization:")
print(words)

#Regex Tokenization
tokenizer = RegexpTokenizer(r'\w+')
print("\nRegex Tokens:",tokenizer.tokenize(text))

#tweet Tokenization
tokenizer = TweetTokenizer()
print("\nTweet Tokenizer:",tokenizer.tokenize(text))

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("\nStemming:")
print(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("\nLemmatization:")
print(lemmatized_words)

# Lemmatization with POS tagging
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatized_with_pos = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
print("\nLemmatization with POS tagging:")
print(lemmatized_with_pos)
















graph = {
    '1': ['2', '7', '8'], '2': ['3', '6'], '3': ['4', '5'], '4': [],
    '5': [], '6': [], '7': [], '8': ['9', '12'], '9': ['10', '11'],
    '10': [], '11': [], '12': []
}

def dfs(graph, node, goal, visited):
    if node not in visited:
        visited.append(node)
        if node == goal or any(dfs(graph, n, goal, visited) for n in graph[node]):
            return True
    return False

visited = []
dfs(graph, '1', '6', visited)
print(visited)







































def solve_n_queens(n):
    solutions = []
    
    def solve(queens, row):
        if row == n:
            solutions.append(list(queens))
            return

        for col in range(n):
            if not any(col == c or abs(row - r) == abs(col - c) for r, c in enumerate(queens)):
                queens.append(col)
                solve(queens, row + 1)
                queens.pop()

    solve([], 0)
    return solutions

def print_solution(solution, n):
    for r in range(n):
        print(" ".join("Q" if c == solution[r] else "." for c in range(n)))
    print()

if __name__ == "__main__":
    N = 8 
    solutions = solve_n_queens(N)
    print(f"Total solutions for N={N}: {len(solutions)}\n")

    for i, solution in enumerate(solutions, 1):
        print(f"Solution {i}:")
        print_solution(solution, N)










































GRAPH = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
    'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

def ucs(source, destination):
    pq = [(0, [source])]
    visited_costs = {}

    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited_costs and visited_costs[node] < cost:
            continue
        
        if node == destination:
            return cost, path

        visited_costs[node] = cost

        for neighbor, weight in GRAPH.get(node, {}).items():
            new_path = path + [neighbor]
            new_cost = cost + weight
            heapq.heappush(pq, (new_cost, new_path))
            
    return float('inf'), []

if __name__ == '__main__':
    source = input('ENTER SOURCE: ').strip()
    goal = input('ENTER GOAL: ').strip()

    if source not in GRAPH or goal not in GRAPH:
        print('ERROR: CITY DOES NOT EXIST.')
    else:
        cost, path = ucs(source, goal)
        print('\nCHEAPEST PATH:')
        print('PATH COST =', cost)
        print(' -> '.join(path))







































import heapq

GRAPH = {
    'Arad': {'Sibiu': 140, 'Zerind': 75, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
    'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Iasi': 92, 'Urziceni': 142},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

STRAIGHT_LINE = {
    'Arad': 366, 'Zerind': 374, 'Oradea': 380, 'Sibiu': 253, 'Timisoara': 329,
    'Lugoj': 244, 'Mehadia': 241, 'Drobeta': 242, 'Craiova': 160, 'Rimnicu': 193,
    'Fagaras': 176, 'Pitesti': 100, 'Bucharest': 0, 'Giurgiu': 77, 'Urziceni': 80,
    'Hirsova': 151, 'Eforie': 161, 'Vaslui': 199, 'Iasi': 226, 'Neamt': 234
}

def a_star(source, destination):
    pq = [(STRAIGHT_LINE[source], 0, [source])]
    visited = set()

    while pq:
        heuristic, cost, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited:
            continue
        
        if node == destination:
            return STRAIGHT_LINE[destination], cost, path
        
        visited.add(node)
        
        for neighbor, weight in GRAPH.get(node, {}).items():
            if neighbor not in visited:
                new_cost = cost + weight
                new_heuristic = new_cost + STRAIGHT_LINE[neighbor]
                heapq.heappush(pq, (new_heuristic, new_cost, path + [neighbor]))
    
    return float('inf'), float('inf'), []

if __name__ == '__main__':
    source = input('ENTER SOURCE: ').strip()
    goal = input('ENTER GOAL: ').strip()

    if source not in GRAPH or goal not in GRAPH:
        print('ERROR: CITY DOES NOT EXIST.')
    else:
        heuristic, cost, path = a_star(source, goal)
        print('\nOPTIMAL PATH:')
        print('HEURISTIC =', heuristic)
        print('PATH COST =', cost)
        print(' -> '.join(path))








































import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

text = "The striped bats are hanging on their feet for best"
words = nltk.word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_words = [stemmer.stem(word) for word in words]

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]

print("Original Text: ", text)
print("Tokenized Words: ", words)
print("Stemmed Words: ", stemmed_words)
print("Lemmatized Words: ", lemmatized_words)

