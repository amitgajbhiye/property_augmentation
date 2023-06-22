# McRae Analysis finding the commonality in concepts based on Cnetp+ChatGPT properties

import pandas as pd
import pickle


def cluster_overlap():
    mcrae_main_cluster = "trained_models/mcrae_analysis_exp/filter_model_dberta_cslb/main_cluster_deb_cslb_filtered_filterthresh0.75.tsv"
    cc_cluster = pd.read_csv(
        mcrae_main_cluster,
        sep="\t",
        header=None,
        names=["concept", "property", "filter_thresh", "concept2count"],
    )
    cc_cluster = cc_cluster[["concept", "property"]]

    cc_cluster["property"] = cc_cluster["property"].apply(str.lower)
    cc_cluster["concept"] = cc_cluster["concept"].apply(str.lower)

    cnetchat_num_clusters = cc_cluster["property"].nunique()
    print("Clustered Df", flush=True)
    print(cc_cluster, flush=True)

    print(flush=True)
    print(f"cnetchat_num_clusters : {cnetchat_num_clusters}", flush=True)

    all_train_test_file = "data/mcrae_analysis/mcrae_train_test_data.tsv"
    mc_train_test = pd.read_csv(
        all_train_test_file, sep="\t", names=["concept", "property", "label"]
    )

    mc_train_test["property"] = mc_train_test["property"].apply(str.lower)
    mc_train_test["concept"] = mc_train_test["concept"].apply(str.lower)

    print(f"McRae All train Test Dataset", flush=True)
    print(mc_train_test, flush=True)

    mcrae_num_clusters = mc_train_test["property"].nunique()
    print(flush=True)
    print(f"mcrae_num_clusters : {mcrae_num_clusters}", flush=True)

    con_overlap_dict = {}
    no_overlap_prop_pair = []

    for cc_prop in cc_cluster["property"].unique():
        for mc_prop in mc_train_test["property"].unique():
            cc_con_cluster = set(
                cc_cluster[cc_cluster["property"] == cc_prop]["concept"]
            )
            mc_con_cluster = set(
                mc_train_test[
                    (mc_train_test["property"] == mc_prop)
                    & (mc_train_test["label"] == 1)
                ]["concept"]
            )

            concept_overlap = cc_con_cluster.intersection(mc_con_cluster)

            if concept_overlap:
                print("*" * 80, flush=True)
                print(
                    f"***cnetp_chatgpt_prop, mc_prop : {cc_prop, mc_prop}", flush=True
                )
                print(
                    f"***cnetp_chatgpt_prop_con_cluster : {len(cc_con_cluster)} {cc_con_cluster}",
                    flush=True,
                )
                print(
                    f"***mc_prop_con_cluster : {len(mc_con_cluster)}, {mc_con_cluster}",
                    flush=True,
                )
                print(
                    f"***concept_overlap : {len(concept_overlap)}, {concept_overlap}",
                    flush=True,
                )
                print(flush=True)

                intersection_count = len(cc_con_cluster.intersection(mc_con_cluster))
                union_count = len(cc_con_cluster.union(mc_con_cluster))

                j_score = round(float(intersection_count) / union_count, 5)
                inclusion_cc_prop = float(intersection_count) / len(cc_con_cluster)
                inclusion_mc_prop = float(intersection_count) / len(mc_con_cluster)

                # con_overlap_dict[(cc_prop, mc_prop)] = len(concept_overlap)

                con_overlap_dict[(cc_prop, mc_prop)] = (
                    j_score,
                    len(concept_overlap),
                    inclusion_cc_prop,
                    inclusion_mc_prop,
                )

            else:
                no_overlap_prop_pair.append((cc_prop, mc_prop))

    sorted_con_overlap_list = sorted(
        con_overlap_dict.items(), key=lambda x: x[1][1], reverse=True
    )

    top_200_clusters = sorted_con_overlap_list[0:200]

    print(f"top_200_clusters : {top_200_clusters}", flush=True)

    top_cluster_file = "trained_models/mcrae_analysis_exp/filter_model_dberta_cslb/top_200_overlap_count_con_overlap_between_cnetpchatp_prop_clusters_mcrae_prop_cluster.txt"

    with open(top_cluster_file, "w") as outfile:
        for (cc_prop, mc_prop), (
            j_score,
            count,
            inclusion_cc_prop,
            inclusion_mc_prop,
        ) in top_200_clusters:
            cc_con_cluster = set(
                cc_cluster[cc_cluster["property"] == cc_prop]["concept"]
            )

            mc_con_cluster = set(
                mc_train_test[
                    (mc_train_test["property"] == mc_prop)
                    & (mc_train_test["label"] == 1)
                ]["concept"]
            )

            con_overlap = cc_con_cluster.intersection(mc_con_cluster)

            extra_cons_cc_prop_cluster = [
                con for con in cc_con_cluster if con not in mc_con_cluster
            ]

            outfile.write(f'{"*" * 80}\n')
            outfile.write(f"***j_score: {j_score}\n")
            outfile.write(f"***overlap_count: {count}\n")
            outfile.write(f"***inclusion_cc_prop: {inclusion_cc_prop}\n")
            outfile.write(f"***inclusion_mc_prop: {inclusion_mc_prop}\n")
            outfile.write(f"***cc_prop, mc_prop: {(cc_prop, mc_prop)}\n")
            outfile.write(
                f"***cc_con_cluster: {len(cc_con_cluster)}, {cc_con_cluster}\n"
            )
            outfile.write(
                f"***mc_con_cluster: {len(mc_con_cluster)}, {mc_con_cluster}\n"
            )
            outfile.write(f"***con_overlap: {len(con_overlap)}, {con_overlap}\n")
            outfile.write(
                f"***extra_cons_cc_prop_cluster: {len(extra_cons_cc_prop_cluster)}, {extra_cons_cc_prop_cluster}\n"
            )
            outfile.write("\n")

    out_file_name = "trained_models/mcrae_analysis_exp/filter_model_dberta_cslb/con_overlap_between_cnetpchatpclusters_mcrae_prop_cluster_list.pkl"
    with open(out_file_name, "wb") as pkl_file:
        pickle.dump(sorted_con_overlap_list, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

    print(f"count_dict_saved_to : {out_file_name}")
    print("sorted_con_overlap_list", flush=True)
    print(sorted_con_overlap_list, flush=True)

    no_overlap_prop_pair_file = "trained_models/mcrae_analysis_exp/filter_model_dberta_cslb/no_overlap_prop_pair_list.pkl"

    with open(no_overlap_prop_pair_file, "wb") as pkl_file:
        pickle.dump(no_overlap_prop_pair, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

    print(f"no_overlap_prop_pair_file : {no_overlap_prop_pair_file}", flush=True)


if __name__ == "__main__":
    cluster_overlap()
