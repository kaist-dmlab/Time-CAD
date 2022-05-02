import time
from datetime import datetime

import firebase_admin
import pandas as pd
from firebase_admin import credentials, firestore


class Firestore:
    def __init__(self):
        self.collection_path = 'data'
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(credentials.Certificate("time-cad-firebase-adminsdk-prm3h-81e63119b0.json"))
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
    firestore.simulate()
    # firestore.get_full_data('timestamp', end=datetime.now())
