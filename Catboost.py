import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import gc
gc.enable()
import warnings
warnings.filterwarnings('ignore')




file = "dataset/transformed_dataset.csv"
reg_cols = ['atlas_pct_diabetes_adults13',
 'atlas_pct_wic15',
 'total_physician_office_net_paid_pmpm_cost_9to12m_b4',
 'atlas_pct_laccess_hisp15',
 'atlas_pct_fmrkt_frveg16',
 'credit_hh_nonmtgcredit_60dpd',
 'atlas_dirsales_farms12',
 'rx_nonmaint_pmpm_ct',
 'zip_cd',
 'atlas_pct_laccess_white15',
 'credit_hh_bankcard_severederog',
 'atlas_pct_fmrkt_credit16',
 'credit_bal_autofinance_new',
 'rej_days_since_last_clm',
 'rx_generic_pmpm_ct_0to3m_b4',
 'rwjf_social_associate_rate',
 'med_physician_office_ds_clm_6to9m_b4',
 'atlas_totalocchu',
 'atlas_veg_acrespth12',
 'atlas_pct_loclsale12',
 'atlas_pct_fmrkt_anmlprod16',
 'atlas_freshveg_farms12',
 'rwjf_resident_seg_black_inx',
 'atlas_pct_loclfarm12',
 'total_outpatient_mbr_resp_pmpm_cost_6to9m_b4',
 'atlas_berry_acrespth12',
 'rx_maint_pmpm_ct_9to12m_b4',
 'rx_tier_2_pmpm_ct',
 'atlas_agritrsm_rct12',
 'atlas_pct_laccess_snap15',
 'atlas_deep_pov_all',
 'ccsp_227_pct',
 'bh_outpatient_net_paid_pmpm_cost',
 'atlas_veg_farms12',
 'rx_hum_16_pmpm_ct',
 'cms_risk_adjustment_factor_a_amt',
 'atlas_recfac14',
 'total_physician_office_copay_pmpm_cost',
 'atlas_pc_fsrsales12',
 'atlas_pct_fmrkt_baked16',
 'atlas_net_international_migration_rate',
 'rx_maint_mbr_resp_pmpm_cost_6to9m_b4',
 'rx_generic_pmpm_cost_6to9m_b4',
 'rx_gpi2_49_pmpm_cost_0to3m_b4',
 'atlas_pct_sbp15',
 'atlas_pct_laccess_child15',
 'met_obe_diag_pct',
 'atlas_orchard_acrespth12',
 'atlas_pct_laccess_hhnv15',
 'cnt_cp_webstatement_pmpm_ct',
 'atlas_pct_laccess_lowi15',
 'rx_gpi2_02_pmpm_cost',
 'cms_partd_ra_factor_amt',
 'atlas_pct_free_lunch14',
 'rx_tier_2_pmpm_ct_3to6m_b4',
 'cons_chva',
 'atlas_pct_fmrkt_wiccash16',
 'rx_overall_net_paid_pmpm_cost_6to9m_b4',
 'total_med_allowed_pmpm_cost_9to12m_b4',
 'bh_physician_office_copay_pmpm_cost_6to9m_b4',
 'atlas_pct_snap16',
 'atlas_ghveg_sqftpth12',
 'atlas_pc_dirsales12',
 'atlas_pct_reduced_lunch14',
 'ccsp_236_pct',
 'atlas_deep_pov_children',
 'atlas_pct_sfsp15',
 'rwjf_air_pollute_density',
 'rx_generic_pmpm_cost',
 'cms_tot_partd_payment_amt',
 'cons_nwperadult',
 'rx_days_since_last_script',
 'atlas_pct_laccess_nhasian15',
 'rx_nonbh_mbr_resp_pmpm_cost_6to9m_b4',
 'rx_days_since_last_script_6to9m_b4',
 'atlas_pct_obese_adults13',
 'credit_bal_consumerfinance',
 'atlas_pct_fmrkt_wic16',
 'atlas_orchard_farms12',
 'atlas_berry_farms12',
 'atlas_pct_laccess_multir15',
 'rx_bh_mbr_resp_pmpm_cost_9to12m_b4',
 'atlas_pc_wic_redemp12',
 'rwjf_mv_deaths_rate',
 'atlas_povertyunder18pct',
 'rx_gpi2_72_pmpm_cost_6to9m_b4',
 'atlas_pct_fmrkt_snap16',
 'atlas_medhhinc',
 'rx_nonbh_net_paid_pmpm_cost',
 'credit_bal_bankcard_severederog',
 'bh_ip_snf_net_paid_pmpm_cost',
 'atlas_pc_snapben15',
 'rx_nonbh_pmpm_ct_0to3m_b4',
 'rx_overall_mbr_resp_pmpm_cost_0to3m_b4',
 'auth_3mth_post_acute_mean_los',
 'rx_branded_mbr_resp_pmpm_cost',
 'rx_tier_1_pmpm_ct_0to3m_b4',
 'bh_ncdm_pct',
 'atlas_naturalchangerate1016',
 'rx_mail_mbr_resp_pmpm_cost_0to3m_b4',
 'credit_bal_autobank',
 'rx_nonotc_dist_gpi6_pmpm_ct',
 'cons_cgqs',
 'rx_overall_gpi_pmpm_ct_0to3m_b4',
 'credit_hh_bankcardcredit_60dpd',
 'rx_gpi2_01_pmpm_cost_0to3m_b4',
 'cci_dia_m_pmpm_ct',
 'atlas_pct_nslp15',
 'mcc_end_pct',
 'atlas_pct_laccess_black15',
 'credit_bal_mtgcredit_new',
 'credit_hh_1stmtgcredit',
 'cons_chmi',
 'rwjf_income_inequ_ratio',
 'atlas_pct_laccess_pop15',
 'atlas_pc_ffrsales12',
 'atlas_hh65plusalonepct',
 'atlas_pct_fmrkt_sfmnp16',
 'auth_3mth_acute_mean_los',
 'rx_hum_28_pmpm_cost',
 'atlas_pct_laccess_nhna15',
 'atlas_povertyallagespct',
 'rx_nonbh_mbr_resp_pmpm_cost',
 'rx_nonmaint_mbr_resp_pmpm_cost_9to12m_b4',
 'atlas_pct_fmrkt_otherfood16',
 'lab_dist_loinc_pmpm_ct',
 'rx_generic_mbr_resp_pmpm_cost',
 'atlas_pct_laccess_seniors15',
 'atlas_pct_cacfp15',
 'total_outpatient_allowed_pmpm_cost_6to9m_b4',
 'rx_nonmaint_mbr_resp_pmpm_cost',
 'credit_bal_nonmtgcredit_60dpd',
 'atlas_ownhomepct',
 'rx_overall_mbr_resp_pmpm_cost',
 'atlas_redemp_snaps16',
 'atlas_netmigrationrate1016',
 'atlas_percapitainc',
 'phy_em_px_pct',
 'rx_generic_mbr_resp_pmpm_cost_0to3m_b4']

cat_cols = ['bh_ncdm_ind',
 'auth_3mth_post_acute_inf',
 'rx_maint_net_paid_pmpm_cost_t_9-6-3m_b4',
 'ccsp_065_pmpm_ct',
 'auth_3mth_acute_vco',
 'rx_gpi2_72_pmpm_ct_6to9m_b4',
 'auth_3mth_post_acute_men',
 'rej_total_physician_office_visit_ct_pmpm_0to3m_b4',
 'total_physician_office_net_paid_pmpm_cost_t_9-6-3m_b4',
 'bh_ip_snf_net_paid_pmpm_cost_0to3m_b4',
 'mcc_ano_pmpm_ct_t_9-6-3m_b4',
 'atlas_type_2015_update',
 'atlas_retirement_destination_2015_upda',
 'auth_3mth_post_acute_sns',
 'atlas_hiamenity',
 'cons_ltmedicr',
 'auth_3mth_acute_ccs_086',
 'total_physician_office_mbr_resp_pmpm_cost_t_9-6-3m_b4',
 'auth_3mth_acute_cir',
 'atlas_csa12',
 'total_med_net_paid_pmpm_cost_t_6-3-0m_b4',
 'cons_n2pwh',
 'auth_3mth_snf_post_hsp',
 'auth_3mth_post_acute_inj',
 'med_outpatient_mbr_resp_pmpm_cost_t_9-6-3m_b4',
 'rx_gpi2_56_dist_gpi6_pmpm_ct_3to6m_b4',
 'atlas_low_employment_2015_update',
 'auth_3mth_acute_inf',
 'lab_albumin_loinc_pmpm_ct',
 'rx_gpi2_17_pmpm_cost_t_12-9-6m_b4',
 'cons_rxadhs',
 'cons_mobplus',
 'atlas_foodinsec_child_03_11',
 'lang_spoken_cd',
 'bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4',
 'auth_3mth_post_acute_gus',
 'auth_3mth_acute_cad',
 'rx_maint_pmpm_ct_t_6-3-0m_b4',
 'auth_3mth_acute_ccs_044',
 'cons_hxmioc',
 'med_outpatient_visit_ct_pmpm_t_12-9-6m_b4',
 'med_physician_office_allowed_pmpm_cost_t_9-6-3m_b4',
 'auth_3mth_acute_res',
 'auth_3mth_acute_chf',
 'auth_3mth_acute_ccs_030',
 'auth_3mth_dc_hospice',
 'auth_3mth_acute_neo',
 'atlas_type_2015_recreation_no',
 'hum_region',
 'atlas_ghveg_farms12',
 'rx_maint_net_paid_pmpm_cost_t_12-9-6m_b4',
 'auth_3mth_acute_ccs_048',
 'rx_overall_gpi_pmpm_ct_t_6-3-0m_b4',
 'rx_overall_gpi_pmpm_ct_t_12-9-6m_b4',
 'rx_nonbh_pmpm_ct_t_9-6-3m_b4',
 'mcc_chf_pmpm_ct_t_9-6-3m_b4',
 'auth_3mth_post_acute_chf',
 'auth_3mth_psychic',
 'rx_nonotc_pmpm_cost_t_6-3-0m_b4',
 'auth_3mth_acute_end',
 'atlas_low_education_2015_update',
 'src_div_id',
 'auth_3mth_bh_acute',
 'auth_3mth_acute_ccs_067',
 'atlas_type_2015_mining_no',
 'cons_n2pmr',
 'rx_mail_net_paid_pmpm_cost_t_6-3-0m_b4',
 'rej_med_er_net_paid_pmpm_cost_t_9-6-3m_b4',
 'med_outpatient_deduct_pmpm_cost_t_9-6-3m_b4',
 'rej_med_ip_snf_coins_pmpm_cost_t_9-6-3m_b4',
 'rx_generic_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
 'auth_3mth_dc_home',
 'auth_3mth_acute_bld',
 'auth_3mth_acute_ner',
 'oontwk_mbr_resp_pmpm_cost_t_6-3-0m_b4',
 'rx_gpi2_90_dist_gpi6_pmpm_ct_9to12m_b4',
 'atlas_foodhub16',
 'rx_maint_pmpm_cost_t_6-3-0m_b4',
 'auth_3mth_post_acute_ben',
 'est_age',
 'auth_3mth_post_acute_cer',
 'auth_3mth_acute_ccs_153',
 'auth_3mth_acute_dig',
 'total_ip_maternity_net_paid_pmpm_cost_t_12-9-6m_b4',
 'auth_3mth_post_acute_cad',
 'rx_bh_pmpm_ct_0to3m_b4',
 'rx_nonmail_dist_gpi6_pmpm_ct_t_9-6-3m_b4',
 'atlas_persistentchildpoverty_1980_2011',
 'atlas_slhouse12',
 'atlas_population_loss_2015_update',
 'auth_3mth_acute_ccs_094',
 'auth_3mth_post_acute_ner',
 'auth_3mth_acute_ccs_227',
 'rx_overall_dist_gpi6_pmpm_ct_t_6-3-0m_b4',
 'auth_3mth_acute_trm',
 'auth_3mth_post_acute',
 'auth_3mth_acute_dia',
 'auth_3mth_acute_ccs_043',
 'rx_overall_mbr_resp_pmpm_cost_t_6-3-0m_b4',
 'cms_orig_reas_entitle_cd',
 'auth_3mth_post_acute_end',
 'auth_3mth_acute_can',
 'auth_3mth_acute_ccs_172',
 'auth_3mth_dc_home_health',
 'atlas_hipov_1115',
 'rx_phar_cat_cvs_pmpm_ct_t_9-6-3m_b4',
 'rx_gpi2_62_pmpm_cost_t_9-6-3m_b4',
 'cons_n2phi',
 'auth_3mth_post_acute_hdz',
 'auth_3mth_bh_acute_mean_los',
 'auth_3mth_post_acute_dig',
 'auth_3mth_transplant',
 'rx_mail_mbr_resp_pmpm_cost_t_9-6-3m_b4',
 'auth_3mth_acute_sns',
 'auth_3mth_post_acute_vco',
 'auth_3mth_home',
 'rx_nonbh_net_paid_pmpm_cost_t_6-3-0m_b4',
 'auth_3mth_post_acute_ckd',
 'rx_gpi2_34_dist_gpi6_pmpm_ct',
 'rx_gpi2_33_pmpm_ct_0to3m_b4',
 'auth_3mth_dc_ltac',
 'cons_estinv30_rc',
 'rx_phar_cat_humana_pmpm_ct_t_9-6-3m_b4',
 'auth_3mth_acute_men',
 'auth_3mth_dc_snf',
 'cons_hhcomp',
 'bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4',
 'auth_3mth_acute_inj',
 'total_physician_office_visit_ct_pmpm_t_6-3-0m_b4',
 'mabh_seg',
 'auth_3mth_post_acute_res',
 'auth_3mth_bh_acute_men',
 'auth_3mth_acute_hdz',
 'hedis_dia_hba1c_ge9',
 'auth_3mth_post_acute_trm',
 'auth_3mth_hospice',
 'rx_gpi2_39_pmpm_cost_t_6-3-0m_b4',
 'atlas_vlfoodsec_13_15',
 'auth_3mth_dc_acute_rehab',
 'rx_generic_pmpm_cost_t_6-3-0m_b4',
 'auth_3mth_acute_ccs_154',
 'cons_rxmaint',
 'total_bh_copay_pmpm_cost_t_9-6-3m_b4',
 'rx_nonmaint_dist_gpi6_pmpm_ct_t_12-9-6m_b4',
 'rej_med_outpatient_visit_ct_pmpm_t_6-3-0m_b4',
 'cons_rxadhm',
 'auth_3mth_acute_mus',
 'rx_nonbh_pmpm_cost_t_9-6-3m_b4',
 'rx_days_since_last_script_0to3m_b4',
 'auth_3mth_post_acute_cir',
 'auth_3mth_post_acute_dia',
 'auth_3mth_post_er',
 'auth_3mth_dc_no_ref',
 'bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4',
 'auth_3mth_acute',
 'rx_branded_pmpm_ct_t_6-3-0m_b4',
 'atlas_farm_to_school13',
 'auth_3mth_acute_cer',
 'med_ambulance_coins_pmpm_cost_t_9-6-3m_b4',
 'auth_3mth_acute_gus',
 'rx_gpi4_6110_pmpm_ct',
 'cons_hxwearbl',
 'auth_3mth_ltac',
 'auth_3mth_acute_ckd',
 'bh_ip_snf_net_paid_pmpm_cost_6to9m_b4',
 'sex_cd',
 'days_since_last_clm_0to3m_b4',
 'atlas_perpov_1980_0711',
 'auth_3mth_post_acute_mus',
 'auth_3mth_non_er',
 'bh_ncal_ind',
 'auth_3mth_facility',
 'atlas_foodinsec_13_15',
 'auth_3mth_dc_left_ama',
 'race_cd',
 'bh_ip_snf_admit_days_pmpm_t_9-6-3m_b4',
 'auth_3mth_dc_other',
 'cons_stlnindx',
 'auth_3mth_acute_skn',
 'total_allowed_pmpm_cost_t_9-6-3m_b4',
 'auth_3mth_rehab',
 'bh_urgent_care_copay_pmpm_cost_t_12-9-6m_b4',
 'auth_3mth_dc_custodial',
 'auth_3mth_snf_direct',
 'auth_3mth_acute_ccs_042',
 'bh_ip_snf_net_paid_pmpm_cost_9to12m_b4',
 'bh_ip_snf_net_paid_pmpm_cost_3to6m_b4',
 'rx_maint_pmpm_cost_t_12-9-6m_b4',
 'auth_3mth_post_acute_rsk',
 'rev_cms_ansth_pmpm_ct',
 'cons_cwht']
df = pd.read_csv('dataset/transformed_dataset.csv')
student_id = 2000728661

target = "covid_vaccination"


id = "ID"
X, y = df[reg_cols + cat_cols], df[target]
X, X_t, y, y_t = train_test_split(df[reg_cols + cat_cols], df[target], test_size=.2, random_state=student_id, shuffle=True, stratify=df[target])


max_depth=[16],
learning_rate= [.001, .1]
n_estimators= [600, 700]


m = CatBoostClassifier(random_state = student_id, task_type="GPU", devices='0:1', 
                           eval_metric='AUC', thread_count=1, 
                           cat_features=cat_cols, metric_period=40,
                           od_type='Iter', loss_function="Logloss",
                       max_depth=max_depth[0], learning_rate=learning_rate[1], n_estimators=n_estimators[1])
m.fit(X,
            y=y,
            eval_set=(X_t, y_t),
            verbose=True,
            plot=True, 
            use_best_model=True)


# m.grid_search(params,
#             X,
#             y=y,
#             cv=4,
#             partition_random_seed=student_id,
#             refit=True,
#             shuffle=True,
#             stratified=True,
#             verbose=True,
#             plot=True)
# 

m.save_model('models/catboost.cbm',
           format="cbm",
           export_parameters=None,
           pool=None)

