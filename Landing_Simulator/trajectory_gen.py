import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from forces import calc_acc

data = [] #data list

def generate_trajectory(dt,data): #generate the trajectory of the plane during the landing phase
    # A320 plane values
    a = -0.03  #acceleration value before grounding (m/s²)
    t = 0   #time (s)
    s = 0   #distance (m)
    h = 15.24   #altitude (m)
    v = 63.68   #speed (m/s)
    vs = -3.55  #vertical speed(m/s)
    seg = None  #segment of the landing
    while True: #loop until reach v <= 0
        data.append([t,h,s,v,vs,a,seg]) #append to the dataframe the new state
        t = t + dt  #compute time   
        s = s + v * dt  #compute braking distance
        h = h + vs * dt #compute altitude
        
        if h > 0:   #if altitude > 0, In Final Approach
            v = v + a * dt
            seg = "Final approach"
        else:   #else grounded
            h = 0   #altitude and vetical speed are equal to 0
            vs = 0
            a = calc_acc(v) #compute the new acceleration
            v = v + a * dt  #compute the new ground speed
            seg = "Landed"  #segment equal to landed

            if v <= 0:  #condition for the end of the landing
                break
    
    data = np.array(data)
    df = pd.DataFrame(data, columns= ["t","h","s","v","vs","a","seg"])
    datadict = {
        "t": data[:, 0],
        "h": data[:, 1],
        "s": data[:, 2],
        "v": data[:, 3], 
        "vs": data[:, 4],
        "a": data[:, 5],
        "seg": data[:, 6],
    }
    return datadict,df
data_df = pd.DataFrame()
data,data_df = generate_trajectory(0.1,data) #generate date
print(data)
print(data_df)
data_df.to_excel("export.xlsx") #save data
is_landed = data_df['seg'] == 'Landed'
data_df_landed = data_df[is_landed]
print(data_df_landed)
#plot the graph number 1
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Landing trajectory")
ax[0][0].plot(data["t"],data["h"])
ax[0][0].set_ylabel("Altitude (m)")
ax[0][1].plot(data["t"], data["s"]/1000)
ax[0][1].set_ylabel("Distanse (km)")
ax[1][0].plot(data["t"], data["v"])
ax[1][0].set_ylabel("True airspeed (m/s)")
ax[1][1].plot(data["t"], data["vs"])
ax[1][1].set_ylabel("Vertical rate (m/s)")
ax[0][0].legend()
plt.show()
#plot graph number 2
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.suptitle("Grounded values")
ax[0][0].plot(data_df_landed["t"],data_df_landed["a"])
ax[0][0].set_ylabel("Acceleration (m/s²)")
ax[0][1].plot(data_df_landed["t"], data_df_landed["v"])
ax[0][1].set_ylabel("Speed ground (m/s)")
ax[1][0].plot(data_df_landed["t"], data_df_landed["s"]-data_df_landed["s"].iloc[0])
ax[1][0].set_ylabel("Distance (m)")
ax[0][0].legend()
plt.show()
