import os
import pandas as pd
import numpy as np
from os import listdir
from sklearn.preprocessing import OneHotEncoder

inp_dir_path = "data/ontology_completion/sumo"
out_dir_path = "data/ontology_completion/sumo/clusters_with_one_hot_encoding"


def one_hot_encoder(dir_path):
    # inp_files =  listdir(dir_path)

    inp_files = [
        fname
        for fname in listdir(dir_path)
        if (fname.startswith("main_cluster_filterthresh"))
        or (fname.startswith("complete_cluster_filter"))
    ]
    print(f"inp_files: {inp_files}")

    for inp_file in inp_files:
        abs_path = os.path.join(dir_path, inp_file)
        # print (inp_file)
        # print (abs_path)

        df = pd.read_csv(abs_path, sep="\t")
        df_new = df.iloc[:, [0, 1]]
        df_new.columns = ["concept", "property"]
        df_new["property"] = df_new["property"].astype("category")
        df_new["property_new"] = df_new["property"].cat.codes

        enc = OneHotEncoder()

        enc_data = pd.DataFrame(
            enc.fit_transform(df_new[["property_new"]]).toarray(), dtype=int
        )
        final_df = df_new[["concept", "property"]].join(enc_data)

        out_file_name = os.path.join(out_dir_path, inp_file)

        print(final_df)

        assert final_df.shape[0] == df.shape[0]

        with open(out_file_name, "w") as f:
            for row in final_df.values:
                embedding_str = " ".join(["{:d}".format(item) for item in row[2:]])
                f.write(f"{row[0]}\t{row[1]}\t{embedding_str}\n")


if __name__ == "__main__":
    one_hot_encoder(dir_path=inp_dir_path)
