from os import listdir
from os.path import isfile, join
import sys
import numpy
import cPickle as pickle
def readFile(mypath):
	mailName={}
	for i in range(1,11):
		mypathNew = mypath+str(i)+"/"
		mailName[i]= map(lambda x: mypathNew+x,[f for f in listdir(mypathNew) if isfile(join(mypathNew, f))]) # appending folder name along with the file name
	return mailName


def superDataCreate(mailName):
	superData={}
	for key in mailName:
		# i am iterating over key
		tempDict ={}
		for filename in mailName[key]:
			# i have the filenames, need to put the text and the spam or no spam feature
			spam = 0
			if "spm" in filename:
				spam =1
			# need to read the file
			content =""
			with open(filename) as f:
				content = f.read()

			# store it as an array
			tempDict[filename] = [content, spam]
		superData[key] = tempDict
	return superData    


if __name__ == "__main__":

	mailName=readFile("part")			
	print mailName
	superData = superDataCreate(mailName)
	pickle.dump( superData, open( "loadedData", "wb" ) )
	# will create a dictionary of dictionary now
	
	# FolderName: {FileName: [Text Body, spam=1]}


	