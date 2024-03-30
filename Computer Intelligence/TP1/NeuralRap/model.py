import pronouncing
import markovify
import re
import random
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers import Dense

depth = 4 # depth of the network. changing will require a retrain
maxsyllables = 16 # maximum syllables per line. Change this freely without retraining the network
train_mode = False
artist = "Racionais MCs" # used when saving the trained model
rap_file = "neural_rap.txt" # where the rap is written to

def create_network(depth):
	model = Sequential()
	model.add(LSTM(4, input_shape=(2, 2), return_sequences=True))
	for i in range(depth):
		model.add(LSTM(8, return_sequences=True))
	model.add(LSTM(2, return_sequences=True))
	model.summary()
	model.compile(optimizer='rmsprop', loss='mse')

	if artist + ".weights.h5" in os.listdir(".") and train_mode == False:
		model.load_weights(str(artist + ".weights.h5"))
		print ("loading saved network: " + str(artist) + ".weights.h5") 
	return model

def markov(text_file):
	read = open(text_file, "r", encoding="utf8").read()
	text_model = markovify.NewlineText(read)
	return text_model

def syllables(line):
	count = 0
	for word in line.split(" "):
		vowels = 'aeiouy'
		word = word.lower().strip(".:;?!")
		if word[0] in vowels:
			count +=1
		for index in range(1,len(word)):
			if word[index] in vowels and word[index-1] not in vowels:
				count +=1
		if word.endswith('e'):
			count -= 1
		if word.endswith('le'):
			count+=1
		if count == 0:
			count +=1
	return count / maxsyllables

import os
import re
import pronouncing

def rhymeindex(artist, lyrics, train_mode=False):
    if f"{artist}.rhymes" in os.listdir(".") and not train_mode:
        print(f"Loading saved rhymes from {artist}.rhymes")
        return open(f"{artist}.rhymes", "r", encoding="utf8").read().split("\n")
    else:
        rhyme_master_list = []
        print("Alright, building the list of all the rhymes")
        for line in lyrics:
            word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
            rhymeslist = pronouncing.rhymes(word)
            rhymeslistends = [x[-2:] for x in rhymeslist]
            try:
                rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
            except Exception:
                rhymescheme = word[-2:]
            rhyme_master_list.append(rhymescheme)
        rhyme_master_list = list(set(rhyme_master_list))
        rhyme_master_list.sort()  # Classificar a lista de rimas

        reverselist = [x[::-1] for x in rhyme_master_list]
        rhymelist = sorted(reverselist, reverse=True)

        with open(f"{artist}.rhymes", "w", encoding="utf8") as f:
            f.write("\n".join(rhymelist))
        print(rhymelist)
        return rhymelist

def rhyme(line, rhyme_list):
	word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
	rhymeslist = pronouncing.rhymes(word)
	rhymeslist = [x.encode('UTF8') for x in rhymeslist]
	rhymeslistends = []
	for i in rhymeslist:
		rhymeslistends.append(i[-2:])
	try:
		rhymescheme = max(set(rhymeslistends), key=rhymeslistends.count)
	except Exception:
		rhymescheme = word[-2:]
	try:
		float_rhyme = rhyme_list.index(rhymescheme)
		float_rhyme = float_rhyme / float(len(rhyme_list))
		return float_rhyme
	except Exception:
		return None


def split_lyrics_file(text_file):
	text = open(text_file, encoding="utf8").read()
	text = text.split("\n")
	while "" in text:
		text.remove("")
	return text


def generate_lyrics(text_model, text_file):
	bars = []
	last_words = []
	lyriclength = len(open(text_file, encoding="utf8").read().split("\n"))
	count = 0
	markov_model = markov(text_file)
	
	while len(bars) < lyriclength / 9 and count < lyriclength * 2:
		bar = markov_model.make_sentence()

		if type(bar) != type(None) and syllables(bar) < 1:
			
			def get_last_word(bar):
				last_word = bar.split(" ")[-1]
				if last_word[-1] in "!.?,":
					last_word = last_word[:-1]
				return last_word
				
			last_word = get_last_word(bar)
			if bar not in bars and last_words.count(last_word) < 3:
				bars.append(bar)
				last_words.append(last_word)
				count += 1
	return bars

import numpy as np

def build_dataset(lines, rhyme_list):
    dataset = []
    for line in lines:
        syllable_count = syllables(line)
        rhyme_value = rhyme(line, rhyme_list)
        dataset.append([syllable_count, rhyme_value])

    x_data = []
    y_data = []

    for i in range(len(dataset) - 3):
        line1 = dataset[i]
        line2 = dataset[i + 1]
        line3 = dataset[i + 2]
        line4 = dataset[i + 3]

        x = np.array([line1[0], line1[1], line2[0], line2[1]], dtype=np.float32)
        x = x.reshape(2, 2)
        x_data.append(x)

        y = np.array([line3[0], line3[1], line4[0], line4[1]], dtype=np.float32)
        y = y.reshape(2, 2)
        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

	
def compose_rap(lines, rhyme_list, lyrics_file, model):
    rap_vectors = []
    human_lyrics = split_lyrics_file(lyrics_file)
    
    initial_index = random.choice(range(len(human_lyrics) - 1))
    initial_lines = human_lyrics[initial_index:initial_index + 2]
    
    starting_input = []
    for line in initial_lines:
        starting_input.append([syllables(line), rhyme(line, rhyme_list)])
    
    # Convertendo starting_input para um array numpy e garantindo que seja do tipo float32
    starting_input = np.array(starting_input, dtype=np.float32)
    
    # Reshape a entrada para corresponder ao formato esperado pelo modelo
    starting_input = starting_input.reshape(1, 2, 2)

    # Fazendo a previsão usando o modelo
    starting_vectors = model.predict(starting_input)
    rap_vectors.append(starting_vectors)
    
    for i in range(100):
        # Continuar fazendo previsões usando o último vetor gerado
        rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).reshape(1, 2, 2)))
    
    return rap_vectors
	
def vectors_into_song(vectors, generated_lyrics, rhyme_list):
	print ("\n\n")	
	print( "About to write rap (this could take a moment)...")
	print ("\n\n")
	def last_word_compare(rap, line2):
		penalty = 0 
		for line1 in rap:
			word1 = line1.split(" ")[-1]
			word2 = line2.split(" ")[-1]
			 
			while word1[-1] in "?!,. ":
				word1 = word1[:-1]
			
			while word2[-1] in "?!,. ":
				word2 = word2[:-1]
			
			if word1 == word2:
				penalty += 0.2
				
		return penalty

	def calculate_score(vector_half, syllables, rhyme, penalty):
		desired_syllables = vector_half[0]
		desired_rhyme = vector_half[1]
		desired_syllables *= maxsyllables

		# Verifique se syllables e rhyme são diferentes de None antes de usá-los
		if syllables is not None and rhyme is not None:
			# Verifique se rhyme é uma lista antes de usar len()
			if isinstance(rhyme, list):
				desired_rhyme *= len(rhyme)
				score = 1.0 - (abs(float(desired_syllables) - float(syllables)) + abs(float(desired_rhyme) - float(rhyme))) - penalty
			else:
				# Lidar com o caso em que rhyme não é uma lista
				score = 0.0  # ou outro valor padrão apropriado, dependendo do seu contexto
		else:
			score = 0.0  # ou outro valor padrão apropriado, dependendo do seu contexto

		return score

		
	dataset = []
	for line in generated_lyrics:
		line_list = [line, syllables(line), rhyme(line, rhyme_list)]
		dataset.append(line_list)
	
	rap = []
	
	vector_halves = []
	
	for vector in vectors:
		vector_halves.append(list(vector[0][0])) 
		vector_halves.append(list(vector[0][1]))
		
	for vector in vector_halves:
		scorelist = []
		for item in dataset:
			line = item[0]
			
			if len(rap) != 0:
				penalty = last_word_compare(rap, line)
			else:
				penalty = 0
			total_score = calculate_score(vector, item[1], item[2], penalty)
			score_entry = [line, total_score]
			scorelist.append(score_entry)
		
		fixed_score_list = []
		for score in scorelist:
			fixed_score_list.append(float(score[1]))

		if fixed_score_list:
			max_score = max(fixed_score_list)
			for item in scorelist:
				if item[1] == max_score:
					rap.append(item[0])
					print(str(item[0]))
					for i in dataset:
						if item[0] == i[0]:
							dataset.remove(i)
							break
					break
		else:
			print("A lista de pontuações está vazia.")

	return rap

def train(x_data, y_data, model):
	model.fit(np.array(x_data), np.array(y_data),
			  batch_size=2,
			  epochs=5,
			  verbose=1)
	model.save_weights(artist + ".weights.h5")
			  
def main(depth, train_mode):
	model = create_network(depth)
	text_file = "lyrics.txt"
	text_model = markov(text_file)
	
	if train_mode == True:
		bars = split_lyrics_file(text_file)
	
	if train_mode == False:
		bars = generate_lyrics(text_model, text_file)
	
	rhyme_list = rhymeindex(artist, bars, train_mode)
	if train_mode == True:
		x_data, y_data = build_dataset(bars, rhyme_list)
		train(x_data, y_data, model)

	if train_mode == False:
		vectors = compose_rap(bars, rhyme_list, text_file, model)
		rap = vectors_into_song(vectors, bars, rhyme_list)
		f = open(rap_file, "w", encoding="utf8")
		for bar in rap:
			f.write(bar)
			f.write("\n")
		
main(depth, train_mode)
