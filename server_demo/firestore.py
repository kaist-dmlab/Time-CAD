import time
from datetime import datetime

import firebase_admin
import numpy as np
import pandas as pd
from firebase_admin import credentials, firestore

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)


class Firestore:
    def __init__(self):
        self.collection_path = 'data'
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(credentials.Certificate("time-cad-v2-firebase-adminsdk-9xwe3-437da1ee87.json"))
        self.db = firestore.client()
        self.db_watch = self.db.collection(self.collection_path).on_snapshot(self.on_arrival)

    @staticmethod
    def on_arrival(doc_snapshot, changes, read_time):
        """ Listen to new data added to Firestore.

        :param doc_snapshot: snapshot of the whole data after update
        :param changes: changes made to data between snapshots
        :param read_time: timestamp
        """

        for change in changes:
            if change.type.name == 'ADDED':
                print(f'<=== ARRIVAL: \t{change.document.id}')

    def get_full_data(self, start: datetime = None, end: datetime = None) -> pd.DataFrame:
        """ Read full data from Firestore as dataframe.

        :param start: the starting datetime to query
        :param end: the ending datetime to query
        :return: a pandas dataframe that contains full data
        """
        ref = self.db.collection(self.collection_path)
        if start is not None:
            ref = ref.where('unix_timestamp', '>', start.strftime('%s'))
        if end is not None:
            ref = ref.where('unix_timestamp', '<', end.strftime('%s'))
        docs = ref.stream()
        lst = []
        for doc in docs:
            d = doc.to_dict()
            lst.append(d)
        df = pd.DataFrame(lst)
        print(f"Successful Firestore stream response. Size: {len(df)}")
        print(df.head())
        return df

    def add_data(self, data: dict, id_name: str):
        """ Add data to Firestore in dictionary form.

        :param data: data to add
        :param id_name: the name of key to use as id
        """
        assert id_name in data.keys()

        doc_id = data[id_name]
        # del data[id_name]
        self.db.collection(self.collection_path).document(doc_id).set(data)
        print(f"===> PUSH: \t\t{doc_id}-{data}")

    def add_stats(self, data: dict):
        """ Add main stats data to Firestore in dictionary form.

        :param data: data to add
        """
        doc_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.db.collection('main_stats').document(doc_id).set(data)
        print(f"===> PUSH: \t\t{data}")

    def add_anomaly_details(self, id: str, data: dict):
        """ Add anomaly details data to Firestore in dictionary form.

        :param id: id of the anomaly
        :param data: data to add
        """
        self.db.collection('anomaly_details').document(id).set(data)
        print(f"===> PUSH: \t\t{data}")

    def upload_data(self, path: str, dev=False):
        df = pd.read_csv(path)
        print(len(df))
        if dev:
            anomaly_df = df[df['label'] == 1].sample(n=30)
            print(len(anomaly_df))
            sampled_df = df.sample(n=30)
            df = pd.concat([anomaly_df, sampled_df])
            df = df.sort_values(by='date')
            for col in df.columns:
                if col not in ['date', 'label']:
                    df[f'label_{col}'] = np.random.choice(a=[0, 1], size=(len(df),))
                    df[f'score_{col}'] = np.random.random(size=(len(df),))
            df['score'] = np.random.random(size=(len(df),))

        print(df)
        for row in df.to_dict('records'):
            self.add_data(row, 'date')

    def simulate(self):
        """ Simulate data push and listen."""
        for i in range(100):
            now = datetime.now()
            data = {'unix_timestamp': now.strftime('%s'), 'pretty_timestamp': now.strftime('%b. %d %X'), 'value': i * 100}
            time.sleep(2)
            self.add_data(data, 'unix_timestamp')

        self.db_watch.unsubscribe()


if __name__ == "__main__":
    firestore = Firestore()
    # firestore.simulate()
    firestore.upload_data('A_slice.csv', dev=True)
    # firestore.get_full_data('timestamp', end=datetime.now())
