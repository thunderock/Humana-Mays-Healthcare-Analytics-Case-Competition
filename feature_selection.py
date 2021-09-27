from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import warnings
import gc
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
# import swifter
gc.enable()
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (20, 20)})
student_id = 2000728661

df = pd.read_csv("dataset/2021_Competition_Training.csv", low_memory=False)
# xls = pd.read_excel("../input/humanamays-healthcare-analytics-case-competition/Humana_Mays_2021_DataDictionary.xlsx")
# tdf = pd.read_csv("../input/humanamays-healthcare-analytics-case-competition/2021_Competition_Holdout.csv")
target = "covid_vaccination"

reg_cols, cat_cols = [], []
id = "ID"

for i in df.columns:
	if i not in [id, target]:
		if df[i].nunique() > 100:
			reg_cols.append(i)
		else:
			cat_cols.append(i)

# reg_cols = ["rx_gpi2_72_pmpm_cost_6to9m_b4", "atlas_pct_laccess_child15", "atlas_recfacpth14", "atlas_pct_fmrkt_frveg16", "atlas_pct_free_lunch14", "atlas_pc_snapben15", "credit_bal_nonmtgcredit_60dpd", "rx_bh_mbr_resp_pmpm_cost_9to12m_b4", "atlas_pct_laccess_nhna15", "credit_hh_nonmtgcredit_60dpd", "cons_lwcm10", "atlas_fsrpth14", "atlas_wicspth12", "cmsd2_sns_digest_abdomen_pmpm_ct", "credit_hh_bankcardcredit_60dpd", "total_outpatient_allowed_pmpm_cost_6to9m_b4", "atlas_netmigrationrate1016", "atlas_pct_laccess_snap15", "rx_nonmaint_mbr_resp_pmpm_cost_9to12m_b4", "atlas_naturalchangerate1016", "ccsp_236_pct", "atlas_pct_laccess_hisp15", "rx_overall_mbr_resp_pmpm_cost", "atlas_pct_laccess_hhnv15", "credit_bal_consumerfinance", "rwjf_uninsured_pct", "rx_mail_mbr_resp_pmpm_cost_0to3m_b4", "atlas_pct_wic15", "ccsp_193_pct", "atlas_pct_fmrkt_baked16", "rx_nonmaint_mbr_resp_pmpm_cost", "credit_hh_bankcard_severederog", "rx_hum_16_pmpm_ct", "cnt_cp_webstatement_pmpm_ct", "atlas_pct_laccess_seniors15", "phy_em_px_pct", "atlas_percapitainc", "rwjf_uninsured_adults_pct", "rx_generic_mbr_resp_pmpm_cost_0to3m_b4", "rx_gpi2_02_pmpm_cost", "atlas_pct_sfsp15", "total_physician_office_net_paid_pmpm_cost_9to12m_b4", "atlas_pc_dirsales12", "cms_tot_partd_payment_amt", "rx_nonotc_dist_gpi6_pmpm_ct", "rx_nonmaint_pmpm_ct", "rx_nonbh_mbr_resp_pmpm_cost_6to9m_b4", "rx_nonbh_mbr_resp_pmpm_cost", "atlas_redemp_snaps16", "total_physician_office_copay_pmpm_cost", "atlas_pct_fmrkt_anmlprod16", "credit_num_agencyfirstmtg", "atlas_agritrsm_rct12", "atlas_pct_laccess_pop15", "rx_gpi2_01_pmpm_cost_0to3m_b4", "rwjf_uninsured_child_pct", "credit_bal_mtgcredit_new", "atlas_pct_laccess_nhasian15", "atlas_deep_pov_all", "atlas_net_international_migration_rate", "atlas_deep_pov_children", "bh_ncdm_pct", "rx_branded_mbr_resp_pmpm_cost", "atlas_pc_wic_redemp12", "rwjf_mv_deaths_rate",
#             "atlas_pct_reduced_lunch14", "rx_hum_28_pmpm_cost", "atlas_totalocchu", "atlas_pct_loclfarm12", "rx_generic_mbr_resp_pmpm_cost", "total_outpatient_mbr_resp_pmpm_cost_6to9m_b4", "rx_gpi4_3400_pmpm_ct", "lab_dist_loinc_pmpm_ct", "atlas_pct_nslp15", "atlas_pct_laccess_lowi15", "atlas_pct_fmrkt_sfmnp16", "atlas_pct_loclsale12", "credit_bal_autobank", "rx_overall_mbr_resp_pmpm_cost_0to3m_b4", "rx_nonbh_net_paid_pmpm_cost", "cms_risk_adjustment_factor_a_amt", "rx_generic_pmpm_cost", "credit_num_autofinance", "rx_maint_mbr_resp_pmpm_cost_6to9m_b4", "atlas_pct_laccess_black15", "atlas_hh65plusalonepct", "bh_outpatient_net_paid_pmpm_cost", "rx_generic_pmpm_cost_6to9m_b4", "atlas_convspth14", "total_med_allowed_pmpm_cost_9to12m_b4", "atlas_pc_ffrsales12", "credit_bal_bankcard_severederog", "rx_gpi2_34_pmpm_ct", "atlas_veg_acrespth12", "atlas_grocpth14", "atlas_pct_fmrkt_snap16", "met_obe_diag_pct", "cms_partd_ra_factor_amt", "atlas_pct_sbp15", "rwjf_resident_seg_black_inx", "atlas_pct_cacfp15", "pdc_lip", "atlas_ffrpth14", "credit_num_autobank_new", "rx_tier_2_pmpm_ct", "atlas_berry_acrespth12", "atlas_pct_fmrkt_credit16", "atlas_pc_fsrsales12", "credit_hh_1stmtgcredit", "atlas_pct_fmrkt_wiccash16", "atlas_fmrktpth16", "cci_dia_m_pmpm_ct", "rwjf_income_inequ_ratio", "credit_num_nonmtgcredit_60dpd", "credit_bal_autofinance_new", "rwjf_men_hlth_prov_ratio", "bh_ncal_pct", "atlas_pct_snap16", "ccsp_227_pct", "atlas_ghveg_sqftpth12", "atlas_orchard_acrespth12", "atlas_pct_laccess_multir15", "atlas_medhhinc", "rwjf_mental_distress_pct", "zip_cd", "atlas_pct_laccess_nhpi15", "credit_num_consumerfinance_new", "rx_gpi2_49_pmpm_cost_0to3m_b4", "rx_overall_net_paid_pmpm_cost_6to9m_b4", "atlas_ownhomepct", "atlas_pct_fmrkt_wic16", "rwjf_social_associate_rate", "mcc_end_pct", "cons_lwcm07", "atlas_pct_fmrkt_otherfood16", "atlas_pct_laccess_white15", "rx_gpi2_66_pmpm_ct"]
# cat_cols = ["auth_3mth_post_acute_dia", "bh_ip_snf_net_paid_pmpm_cost_9to12m_b4", "auth_3mth_acute_ckd", "bh_ncal_pmpm_ct", "src_div_id", "total_bh_copay_pmpm_cost_t_9-6-3m_b4", "bh_ip_snf_net_paid_pmpm_cost_3to6m_b4", "cons_chmi", "mcc_ano_pmpm_ct_t_9-6-3m_b4", "auth_3mth_post_acute_trm", "rx_maint_pmpm_cost_t_12-9-6m_b4", "auth_3mth_post_acute_rsk", "cons_ltmedicr", "rx_gpi4_6110_pmpm_ct", "rx_nonbh_pmpm_cost_t_9-6-3m_b4", "auth_3mth_acute_vco", "rx_bh_pmpm_ct_0to3m_b4", "auth_3mth_dc_ltac", "auth_3mth_post_acute_inj", "auth_3mth_dc_home", "rx_gpi2_17_pmpm_cost_t_12-9-6m_b4", "cons_hxmioc", "rx_generic_pmpm_cost_t_6-3-0m_b4", "atlas_ghveg_farms12", "cons_cwht", "bh_ncdm_ind", "atlas_retirement_destination_2015_upda", "rx_overall_mbr_resp_pmpm_cost_t_6-3-0m_b4", "bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4", "rx_overall_dist_gpi6_pmpm_ct_t_6-3-0m_b4", "auth_3mth_post_acute_ben", "auth_3mth_dc_no_ref", "rx_overall_gpi_pmpm_ct_0to3m_b4", "auth_3mth_dc_snf", "rx_phar_cat_humana_pmpm_ct_t_9-6-3m_b4", "auth_3mth_acute_ccs_048", "bh_ip_snf_net_paid_pmpm_cost_0to3m_b4", "auth_3mth_acute_end", "auth_3mth_psychic", "atlas_hiamenity", "auth_3mth_bh_acute", "auth_3mth_acute_chf", "rx_overall_gpi_pmpm_ct_t_6-3-0m_b4", "mcc_chf_pmpm_ct_t_9-6-3m_b4", "bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4", "auth_3mth_hospice", "auth_3mth_acute_bld", "auth_3mth_dc_hospice", "auth_3mth_acute_ccs_030", "auth_3mth_acute_skn", "atlas_veg_farms12", "atlas_vlfoodsec_13_15", "rx_gpi2_34_dist_gpi6_pmpm_ct", "bh_ip_snf_net_paid_pmpm_cost", "est_age", "rx_maint_pmpm_cost_t_6-3-0m_b4", "auth_3mth_acute_neo", "rwjf_air_pollute_density", "atlas_recfac14", "cons_mobplus", "lab_albumin_loinc_pmpm_ct", "atlas_pct_obese_adults13", "rx_maint_net_paid_pmpm_cost_t_12-9-6m_b4", "rev_pm_obsrm_pmpm_ct", "med_ip_snf_admit_days_pmpm", "rej_med_outpatient_visit_ct_pmpm_t_6-3-0m_b4", "auth_3mth_post_acute_vco", "cons_stlnindx", "atlas_hipov_1115", "auth_3mth_post_acute_dig", "atlas_berry_farms12", "rej_med_ip_snf_coins_pmpm_cost_t_9-6-3m_b4", "rwjf_inactivity_pct", "rx_gpi2_72_pmpm_ct_6to9m_b4", "cons_n2pmr", "med_physician_office_allowed_pmpm_cost_t_9-6-3m_b4", "auth_3mth_acute_res", "rev_cms_ct_pmpm_ct", "atlas_foodhub16", "auth_3mth_acute_dig", "auth_3mth_dc_acute_rehab", "auth_3mth_post_acute_hdz", "bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4", "auth_3mth_acute_ccs_172", "total_physician_office_net_paid_pmpm_cost_t_9-6-3m_b4", "auth_3mth_acute_ccs_154", "atlas_type_2015_mining_no", "rx_days_since_last_script", "auth_3mth_post_acute_res", "auth_3mth_acute_inf", "atlas_povertyallagespct", "rx_branded_pmpm_ct_t_6-3-0m_b4", "med_outpatient_deduct_pmpm_cost_t_9-6-3m_b4", "atlas_low_employment_2015_update", "atlas_pct_diabetes_adults13", "auth_3mth_non_er", "atlas_foodinsec_child_03_11", "auth_3mth_acute_cad", "cons_nwperadult", "total_allowed_pmpm_cost_t_9-6-3m_b4", "mabh_seg", "cms_orig_reas_entitle_cd", "med_physician_office_ds_clm_6to9m_b4", "bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4", "auth_3mth_post_acute_cir", "auth_3mth_post_acute_cer", "rx_generic_pmpm_ct_0to3m_b4", "oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4", "bh_ncal_ind", "auth_3mth_post_acute_mus", "hum_region", "rx_nonmail_dist_gpi6_pmpm_ct_t_9-6-3m_b4", "bh_ip_snf_net_paid_pmpm_cost_6to9m_b4", "rej_med_er_net_paid_pmpm_cost_t_9-6-3m_b4",
#             "med_outpatient_mbr_resp_pmpm_cost_t_9-6-3m_b4", "rx_tier_2_pmpm_ct_3to6m_b4", "rx_maint_pmpm_ct_9to12m_b4", "rx_nonbh_net_paid_pmpm_cost_t_6-3-0m_b4", "atlas_type_2015_recreation_no", "auth_3mth_post_acute_sns", "rx_gpi2_39_pmpm_cost_t_6-3-0m_b4", "atlas_type_2015_update", "total_ip_maternity_net_paid_pmpm_cost_t_12-9-6m_b4", "cmsd2_eye_retina_pmpm_ct", "auth_3mth_acute_can", "auth_3mth_post_acute", "auth_3mth_facility", "rx_days_since_last_script_0to3m_b4", "atlas_population_loss_2015_update", "rx_maint_pmpm_ct_t_6-3-0m_b4", "auth_3mth_post_acute_men", "auth_3mth_acute_mean_los", "cons_rxmaint", "rx_mail_net_paid_pmpm_cost_t_6-3-0m_b4", "auth_3mth_home", "cons_hxwearbl", "total_physician_office_mbr_resp_pmpm_cost_t_9-6-3m_b4", "atlas_farm_to_school13", "auth_3mth_acute_inj", "auth_3mth_acute_ccs_153", "rej_days_since_last_clm", "auth_3mth_transplant", "atlas_dirsales_farms12", "rev_cms_ansth_pmpm_ct", "rx_mail_mbr_resp_pmpm_cost_t_9-6-3m_b4", "med_outpatient_visit_ct_pmpm_t_12-9-6m_b4", "rx_nonbh_pmpm_ct_t_9-6-3m_b4", "auth_3mth_acute", "rx_nonbh_pmpm_ct_0to3m_b4", "auth_3mth_dc_left_ama", "atlas_povertyunder18pct", "rx_tier_1_pmpm_ct_0to3m_b4", "auth_3mth_acute_ccs_227", "cons_estinv30_rc", "auth_3mth_bh_acute_men", "auth_3mth_dc_custodial", "total_med_net_paid_pmpm_cost_t_6-3-0m_b4", "rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4", "atlas_csa12", "sex_cd", "rx_gpi2_62_pmpm_cost_t_9-6-3m_b4", "lang_spoken_cd", "rx_overall_gpi_pmpm_ct_t_12-9-6m_b4", "auth_3mth_ltac", "cons_hhcomp", "auth_3mth_acute_hdz", "cons_rxadhs", "auth_3mth_acute_men", "auth_3mth_rehab", "auth_3mth_acute_ccs_086", "cons_n2pwh", "rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4", "atlas_slhouse12", "auth_3mth_snf_post_hsp", "atlas_foodinsec_13_15", "auth_3mth_acute_cer", "cons_rxadhm", "rx_nonotc_pmpm_cost_t_6-3-0m_b4", "auth_3mth_acute_trm", "cons_n2phi", "bh_physician_office_copay_pmpm_cost_6to9m_b4", "rej_total_physician_office_visit_ct_pmpm_0to3m_b4", "auth_3mth_acute_dia", "auth_3mth_snf_direct", "auth_3mth_acute_ccs_067", "auth_3mth_acute_ccs_043", "auth_3mth_dc_home_health", "rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4", "cmsd2_sns_genitourinary_pmpm_ct", "auth_3mth_acute_cir", "auth_3mth_acute_ner", "auth_3mth_acute_ccs_094", "med_ambulance_coins_pmpm_cost_t_9-6-3m_b4", "hedis_dia_hba1c_ge9", "rx_days_since_last_script_6to9m_b4", "atlas_persistentchildpoverty_1980_2011", "auth_3mth_post_acute_cad", "cons_cgqs", "ccsp_065_pmpm_ct", "auth_3mth_acute_ccs_044", "rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4", "bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4", "rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4", "auth_3mth_post_acute_ckd", "auth_3mth_post_acute_ner", "auth_3mth_post_er", "atlas_avghhsize", "atlas_orchard_farms12", "total_physician_office_visit_ct_pmpm_t_6-3-0m_b4", "rx_gpi2_33_pmpm_ct_0to3m_b4", "auth_3mth_post_acute_chf", "atlas_freshveg_farms12", "auth_3mth_acute_ccs_042", "auth_3mth_post_acute_inf", "auth_3mth_acute_sns", "days_since_last_clm_0to3m_b4", "auth_3mth_dc_other", "auth_3mth_bh_acute_mean_los", "auth_3mth_post_acute_gus", "auth_3mth_post_acute_end", "auth_3mth_acute_mus", "atlas_perpov_1980_0711", "auth_3mth_post_acute_mean_los", "auth_3mth_acute_gus", "rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4", "atlas_low_education_2015_update", "race_cd"]

print(df.shape)
print(df.columns)




s = set()
dtypes_null = {}
for col in tqdm(reg_cols):
    dtype = df[col].dtype
    s.add(dtype)
    nulls = df[col].isnull().sum()
    if dtype not in dtypes_null:
        dtypes_null[dtype] = []
    if nulls != 0:
        dtypes_null[dtype].append(nulls / df.shape[0])
print(s)
print(dtypes_null)


s = set()
dtypes_null = {}
for col in tqdm(cat_cols):
    dtype = df[col].dtype
    s.add(dtype)
    nulls = df[col].isnull().sum()
    if dtype not in dtypes_null:
        dtypes_null[dtype] = []
    if nulls != 0:
        dtypes_null[dtype].append(nulls / df.shape[0])
print(s)
print(dtypes_null)


training_cols = reg_cols + cat_cols


# no cat column with int type and null values

label_encoders = {}
for col in tqdm(cat_cols):
    dtype = df[col].dtype
    temp = df[col].fillna('nan').astype(str)
    df[col] = temp


# no reg column with int type and null values
for col in tqdm(reg_cols):
    dtype = df[col].dtype
    # need to scale here, probably
    temp = df[col].fillna(0.)
    df[col] = temp

target_encoder = LabelEncoder()
df.rename(columns={target: target + '_t'})
df[target + "_t"] = target_encoder.fit_transform(df[target])




# plt.figure(figsize=(12, 10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.xticks(rotation=45)
# plt.show()


#Correlation with output variable
# cor_target = abs(cor[target + '_t'])
# #Selecting highly correlated features
# relevant_features = cor_target[cor_target > 0.5]
# print(relevant_features)



#### starting all feature selection stuff
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(df[reg_cols])
constant_columns = [column for column in training_cols if column not in df[reg_cols].columns[constant_filter.get_support()]]

print("size of features before removing constant filters: {}".format(len(training_cols)))
training_cols = list(set(training_cols) - set(constant_columns))

df.to_csv('dataset/transformed_dataset.csv', index=False, columns=[target + '_t'] + training_cols)
print("size of features after removing constant filters: {}".format(len(training_cols)))

print("variances: {}".format(constant_filter.variances_))
X, y = df[training_cols], df[target + '_t']

pca = PCA(n_components=50, random_state=student_id)
fs = SelectKBest(score_func=f_classif, k=100)


combined_features = FeatureUnion([('pca', pca), ('univ_select', fs)], n_jobs=4)

X_features = combined_features.fit(X, y).transform(X)
print("features in: {}".format(combined_features.n_features_in_))
print("Combined space has", X_features.shape[1], "features")

tree = RandomForestClassifier(n_jobs=1, random_state=student_id, max_depth=15)

selection_pipeline = Pipeline([('features', combined_features), ('tree', tree)])

params = dict(
    features__pca__n_components=[25, 50],
    features__univ_select__k=[50, 100],
    tree__n_estimators=[400,600,700])

search = GridSearchCV(selection_pipeline, param_grid=params, verbose=2, n_jobs=2, cv=4, scoring='roc_auc')
search.fit(X, y)

print("-------")
print(search.best_estimator_)
print("-------")
print(search.best_params_)
print("-------")
print(search.cv_results_)
print("-------")
print(search.best_score_)
print("-------")
print(search.best_estimator_.named_steps['features'])
print("-------")
print(fs.feature_names_in_)
print("-------")

joblib.dump(search, 'grid_search_best_tree.pkl')
# joblib.load("model_file_name.pkl")
