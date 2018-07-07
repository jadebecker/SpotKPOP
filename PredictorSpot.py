import h5py
import keras
from keras.models import load_model
import spotipy
import sys
import spotipy.util as util
import numpy as np

######### Recup song a tester ###########

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print ("Usage: %s username" % (sys.argv[0],))
    sys.exit()

x_train = []
songs = []
name = []

scope = 'user-library-read'
token = util.prompt_for_user_token(username, scope)
if token:
	sp = spotipy.Spotify(auth=token)
	results = sp.current_user_saved_tracks(limit=10, offset= 0)
	for item in results['items']:
		track = item['track']
		name.append(track['name'] + ' - ' + track['artists'][0]['name'])
		#	print(sp.audio_features(tracks =[track['id']]))
		songs.append(sp.audio_features(tracks =[track['id']]))
			#x_train = sp.audio_features(tracks =[track['id']]['danceability']['energy'])
else:
    	print ("Can't get token for", username)

model = load_model("Proj_spotify_trop_styleee.h5")

i = 0
for song in songs:
	#x_train.append([song[0]["danceability"],song[0]["loudness"]/(-60),song[0]["energy"],song[0]["valence"],song[0]["duration_ms"]/1000000,song[0]["tempo"]/1000,song[0]["speechiness"],song[0]["instrumentalness"],song[0]["acousticness"],song[0]["liveness"]])
	x_train = ([song[0]["danceability"],song[0]["loudness"]/(-60),song[0]["energy"],song[0]["valence"],song[0]["duration_ms"]/1000000,song[0]["tempo"]/1000,song[0]["speechiness"],song[0]["instrumentalness"],song[0]["acousticness"],song[0]["liveness"]])
######## model ###########################
	x_train = np.array(x_train)
	x_train = x_train.reshape(1,10)
	#print (x_train)
	print (name[i])
	if (np.argmax(model.predict(x_train))==0):
		print("KPOP\n")
	else :
		print("PAS KPOP\n")

	#print(np.argmax(model.predict(x_train)))
	i += 1

#x_train = np.expand_dims(x_train,0)
#print(x_train.shape)

#for element in x_train:
	











