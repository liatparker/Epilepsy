import pandas as pd
import os
output_file = "D:/data/output_1.csv"


def pre_processing(file):
    # The USI dataset is made up of 500 subjects each with 23 seconds divided to 178 of 1/178 seconds
    #  of EEG values labled
    # reading csv file of Epilepsy of USI ,
    # taking all the 1- result and equaliy taking 0 result
    #  for balanced data for the machine learning modouls later

    # epdf = pd.read_csv("D:/data/epii.csv")
    # epdf = epdf.fillna(0)
    # epdf.to_pickle(file)
    epdf = pd.read_pickle(file)

    # looping the subjects (part of 500 subjects , each with 178 of 1/178 seconds for 23 seconds of EEG test )

    for i in range(1, 1000):
        dfi = epdf.loc[epdf['subject'] == i]

        dfi_new = dfi.iloc[:, 0:179]

        # melting
        # to reshape the data
        mei = pd.melt(dfi_new, id_vars=['id'], var_name='Time',
                      value_name='Value')
        # Monitoring the data and  difining as time series
        mei["Time"] = list(map(int, mei["Time"]))

        mei["Time"] = mei["id"].apply(lambda x: (x - (1 / 178))) + mei["Time"].apply(lambda x: (x / 178))
        mei['Time'] = pd.to_datetime(mei['Time'], unit='s')
        mei = mei.set_index('Time')

        if len(mei) > 0:
            tsi = pd.Series(mei.Value, mei.index)
            tsi = tsi.sort_index()
     # creating a new dataframe of std and mean
            Dataframei = pd.DataFrame(tsi).resample("s").mean()
            Dataframei.columns.values[0] = "mean"
            Dataframei['std'] = pd.DataFrame(tsi).resample("s").std()

    # the target

            if dfi['seizure'].all() == 0:
                Dataframei['seizure'] = 0
            else:
                Dataframei['seizure'] = 1
    #  the subject
            Dataframei['subject'] = i

    # creating scv and a pickle file for the process of machine learning , including only mean annd std of each second

            Dataframei.to_csv(output_file, index=False, mode='a', header=(not os.path.exists(output_file)))
            # df = pd.read_csv(output_file)
            # df.to_pickle("D:/data/output_3.pkl")
            # df = pd.read_pickle("D:/data/output_1.pkl")




