from firebase import Firebase


Config = {
    "apiKey": "AIzaSyCYKfiFWKCFam6zzVmjCMpNNSaj-YLlLMA",
    "authDomain": "pcb-fault-detection-database.firebaseapp.com",
    "databaseURL": "https://pcb-fault-detection-database-default-rtdb.firebaseio.com",
    "projectId": "pcb-fault-detection-database",
    "storageBucket": "pcb-fault-detection-database.appspot.com",
    "messagingSenderId": "355750181058",
    "appId": "1:355750181058:web:ae27b9e779d3a68a70235b",
    "measurementId": "G-YZCKMZJDV7"
}

firebase = Firebase(Config)
database = firebase.database()
storage = firebase.storage()


# all_users = database.child("requests").get()
#
# requestslistkey = []
# requestslistvalue = []
#
# datatype=""
# imageurl=""
# push_key=""
# datadic={}
#
# for user in all_users.each():
#     requestslistkey.append(user.key())
#     requestslistvalue.append(user.val())
#
# # print(requestslistkey)
# # print(requestslistvalue)
#
# def stream_handler(message):
#
#     push_key = message["path"].split("/")[-1]
#     datadic = message["data"]
#
#     print(push_key)
#     print(datadic)
#     print(len(datadic))
#
#     if len(datadic)==2:
#         app.main(push_key,datadic)
#         datadic.clear()
#
#
# my_stream1 = database.child("requests").stream(stream_handler)

def setfirebasedata(id,data):

    storage.child(f'{id}/{data["Defect_Detected_Image"]}').put(data["Defect_Detected_Image"])
    storage.child(f'{id}/{data["Colour_Filtered_Image"]}').put(data["Colour_Filtered_Image"])
    storage.child(f'{id}/{data["Classified_Image"]}').put(data["Classified_Image"])

    data["Defect_Detected_Image"] =  storage.child(f'{id}/{data["Defect_Detected_Image"]}').get_url(None)
    data["Colour_Filtered_Image"] =  storage.child(f'{id}/{data["Colour_Filtered_Image"]}').get_url(None)
    data["Classified_Image"]=  storage.child(f'{id}/{data["Classified_Image"]}').get_url(None)
    
    database.child(id).set(data)
    print("Successfully Updated in the Firebase Database")
