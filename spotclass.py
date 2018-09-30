# spotify #
import sys
import spotipy
import spotipy.util as util
# machine learning #
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###### recup nom utilisateur

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print ("Usage: %s username" % (sys.argv[0],))
    sys.exit()

### declaration ####

song_train_id = []
song_train_id_kpop = []

x_train = []
y_train = []

x_test = []
y_test = []

############  CHANSONS PAS KPOP  ############################################

#######################
####  recup songs #####
#######################


for nb_song in tqdm(range(0,900,100)):
	scope = 'user-library-read'
	token = util.prompt_for_user_token(username, scope)
	if token:
		sp = spotipy.Spotify(auth=token)
		results = sp.user_playlist_tracks(user = username,  playlist_id='2tKrzVfEIbNBjiX7Xwu1w7', fields=None, limit=100, offset=0, market=None)
		for item in results['items']:
			track = item['track']
		#	print (track['name'] + ' - ' + track['artists'][0]['name'])
		#	print(sp.audio_features(tracks =[track['id']]))
			song_train_id.append(sp.audio_features(tracks =[track['id']]))
	else:
        	print ("Can't get token for", username)


#########################################
### prendre bon params + entre 0 et 1 ###
#########################################

for songs in tqdm(song_train_id):
	x_train.append([songs[0]["danceability"],songs[0]["loudness"]/(-60),songs[0]["energy"],songs[0]["valence"],songs[0]["duration_ms"]/1000000,songs[0]["tempo"]/1000,songs[0]["speechiness"],songs[0]["instrumentalness"],songs[0]["acousticness"],songs[0]["liveness"]])
	y_train.append(1)

print (len(x_train))

########## KPOP CHANSONS ############################################

#######################
####  recup songs #####
#######################

for nb_song in tqdm(range(0,900,100)):
	scope = 'user-library-read'
	token = util.prompt_for_user_token(username, scope)
	if token:
		sp = spotipy.Spotify(auth=token)
		results = sp.user_playlist_tracks(user = username, playlist_id='3OGDPCKLW0bNYgM2hUPWmf', fields=None, limit=100, offset=0, market=None)
		for item in results['items']:
			track = item['track']
			#print (track['name'] + ' - ' + track['artists'][0]['name'])
			#print(sp.audio_features(tracks =[track['id']]))
			song_train_id_kpop.append(sp.audio_features(tracks =[track['id']]))
	else:
        	print ("Can't get token for", username)


#########################################
### prendre bon params + entre 0 et 1 ###
#########################################

for songsk in tqdm(song_train_id_kpop):
	x_train.append([songsk[0]["danceability"],songsk[0]["loudness"]/(-60),songsk[0]["energy"],songsk[0]["valence"],songsk[0]["duration_ms"]/1000000,songsk[0]["tempo"]/1000,songsk[0]["speechiness"],songsk[0]["instrumentalness"],songsk[0]["acousticness"],songsk[0]["liveness"]])
	y_train.append(0)

print (len(x_train))


############### melange donnees + split ###############

x_train,y_train = shuffle(x_train,y_train)
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size = 0.2)



################## MODEL ####################################################

x_train=np.array(x_train)
x_test=np.array(x_test)
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(1440,10)
x_test = x_test.reshape(360,10)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = keras.utils.to_categorical(y_train,2)
y_test = keras.utils.to_categorical(y_test,2)

#######################
####  def model #####
#######################


model = Sequential()
model.add(Dense(32,input_shape=(10,)))
model.add(LeakyReLU(0.2))
model.add(Dense(64))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(LeakyReLU(0.2))
model.add(Dense(256))
model.add(LeakyReLU(0.2))
model.add(Dense(784))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.25))
model.add(Dense(1024))
model.add(LeakyReLU(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile( loss = 'binary_crossentropy',
			optimizer = Adam(),
			metrics =['accuracy'])


#######################
####  train + test ####
#######################

model.fit(x_train,y_train,
		batch_size = 50,
		epochs = 50,
		validation_data = (x_test, y_test))

#######################
####     save     #####
#######################


model.save("Proj_spotify_trop_styleee.h5")
