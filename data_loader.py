import pandas as pd
import numpy as np

def get_data():
    np.random.seed(42)
    N = 500
    age = np.random.normal(45, 17, N).clip(18, 85).round(0)
    sex = np.random.choice([1, 2], N)
    bmi = np.random.normal(28.5, 6.5, N).clip(16, 55).round(1)
    sbp = (110 + 0.4*age + 0.3*bmi + np.random.normal(0,14,N)).clip(80,210).round(0)
    tchol = (160 + 0.4*age + 0.5*bmi + np.random.normal(0,35,N)).clip(100,360).round(0)
    hdl = (70 - 0.15*bmi - 0.05*age + np.random.normal(0,14,N)).clip(20,120).round(0)
    gluc = (75 + 0.3*age + 0.8*bmi + np.random.normal(0,22,N)).clip(60,350).round(0)
    hba1c = (4.2 + 0.008*gluc + np.random.normal(0,0.6,N)).clip(4.0,13.0).round(1)
    creat = (0.6 + 0.008*age + 0.004*bmi + np.random.normal(0,0.25,N)).clip(0.4,5.0).round(2)
    fev1 = (4.2 - 0.025*age + np.random.normal(0,0.7,N)).clip(0.5,6.5).round(2)
    fvc = (fev1 + np.random.normal(0.7,0.3,N)).clip(fev1,7.5).round(2)
    curr_smk = np.random.choice([1,2,3],N,p=[0.20,0.10,0.70])
    cpd = np.where(curr_smk==1, np.random.choice([5,10,15,20],N).astype(float), 0.0)
    age_s = np.where(curr_smk!=3, np.random.randint(14,28,N).astype(float), np.nan)
    rec_min = np.random.choice([0,30,60,120],N).astype(float)
    drinks = np.random.choice([0,1,2,3,4],N)
    pov = np.random.exponential(2.5,N).clip(0.1,5.0).round(2)
    food_s = np.where(pov<1.0, np.random.choice([3,4],N), np.random.choice([1,2,3,4],N,p=[0.65,0.15,0.12,0.08]))
    insur = np.where(pov<1.3, np.random.choice([1,2],N,p=[0.55,0.45]), np.random.choice([1,2],N,p=[0.88,0.12]))
    is_fem = (sex==2).astype(int)
    is_smk = np.isin(curr_smk,[1,2]).astype(int)
    tc_hdl = (tchol/hdl).round(2)
    yrs_smk = np.where(~np.isnan(age_s),(age-age_s).clip(0),0)
    pack_yrs = ((cpd/20)*yrs_smk).round(1)
    fev1_fvc = (fev1/fvc).round(3)
    kappa = np.where(sex==2,0.7,0.9)
    alpha = np.where(sex==2,-0.329,-0.411)
    sf = np.where(sex==2,1.018,1.0)
    ratio = creat/kappa
    egfr = np.round(141*np.where(ratio<1,ratio**alpha,ratio**-1.209)*(0.9938**age)*sf,1)
    sdoh = ((pov<1.3).astype(int)+np.isin(food_s,[3,4]).astype(int)+(insur==2).astype(int))
    mult = pd.Series(sdoh).map({0:1.0,1:1.08,2:1.18,3:1.32}).values
    cvd = np.random.uniform(0,100,N).round(1)
    t2d = np.random.choice([1,4,17,33,50],N,p=[0.50,0.38,0.09,0.03,0.004]).astype(float)
    resp = np.random.uniform(0,60,N).round(1)
    cancer = (age*0.4+pack_yrs*0.5+np.random.normal(0,8,N)).clip(0,100).round(1)
    comp = (cvd*0.35+t2d*0.25+cancer*0.25+resp*0.15).round(1)
    def tier(s):
        if s<10: return "Low"
        elif s<25: return "Moderate"
        elif s<45: return "High"
        else: return "Very High"
    driver_pool = ["age","is_female","bmi","sbp","tc_hdl_ratio","fasting_glucose","hba1c","is_smoker","pack_years","sdoh_burden","fev1_fvc","vigorous_rec_min"]
    patients = pd.DataFrame({
        "patient_id":np.arange(1,N+1),"age":age,"sex":sex,"is_female":is_fem,
        "bmi":bmi,"waist_cm":(bmi*2.8+np.random.normal(0,8,N)).clip(55,145).round(1),
        "sbp":sbp,"dbp":(65+0.2*age+0.2*bmi+np.random.normal(0,9,N)).clip(50,130).round(0),
        "total_cholesterol":tchol,"hdl":hdl,"ldl":(tchol*0.6-0.1*hdl+np.random.normal(0,25,N)).clip(40,280).round(0),
        "tc_hdl_ratio":tc_hdl,"fasting_glucose":gluc,"hba1c":hba1c,
        "creatinine":creat,"egfr":egfr,"fev1_fvc":fev1_fvc,
        "is_smoker":is_smk,"pack_years":pack_yrs,"vigorous_rec_min":rec_min,
        "past_year_drinks":drinks,"poverty_ratio":pov,"sdoh_burden":sdoh,
        "risk_cvd_10yr":cvd,"risk_cvd_10yr_adj":(cvd*mult).clip(0,100).round(1),
        "risk_t2d_10yr":t2d,"risk_t2d_10yr_adj":(t2d*mult).clip(0,100).round(1),
        "risk_respiratory":resp,"risk_respiratory_adj":(resp*mult).clip(0,100).round(1),
        "risk_cancer":cancer,"risk_cancer_adj":(cancer*mult).clip(0,100).round(1),
        "composite_risk":comp,
        "risk_tier":pd.Series(comp).apply(tier),
        "top_risk_drivers":[str(list(np.random.choice(driver_pool,3,replace=False))) for _ in range(N)],
    })
    habit_templates = {
        "Low":      [{"habit":"Walk 20 minutes daily","description":"Take a brisk 20-min walk each day.","target_risk":"cardiovascular health","difficulty":"Easy","daily_time_minutes":20,"expected_impact":"Improves heart health and metabolism."},
                     {"habit":"Drink more water","description":"Drink 8 glasses of water daily.","target_risk":"general health","difficulty":"Easy","daily_time_minutes":5,"expected_impact":"Supports kidney function."},
                     {"habit":"Sleep 7-8 hours","description":"Set a consistent bedtime.","target_risk":"metabolic health","difficulty":"Medium","daily_time_minutes":0,"expected_impact":"Reduces cortisol and inflammation."}],
        "Moderate": [{"habit":"Reduce sodium intake","description":"Limit processed foods and salt.","target_risk":"blood pressure","difficulty":"Medium","daily_time_minutes":10,"expected_impact":"Lowers blood pressure risk."},
                     {"habit":"Walk 30 minutes daily","description":"Walk briskly for 30 minutes.","target_risk":"cardiovascular health","difficulty":"Easy","daily_time_minutes":30,"expected_impact":"Reduces CVD risk by 35%."},
                     {"habit":"Eat more vegetables","description":"Add 2 servings of vegetables per meal.","target_risk":"cholesterol","difficulty":"Easy","daily_time_minutes":10,"expected_impact":"Lowers LDL cholesterol."}],
        "High":     [{"habit":"30-minute brisk walking","description":"Walk at a moderate-high pace for 30 minutes.","target_risk":"cardiovascular disease","difficulty":"Medium","daily_time_minutes":30,"expected_impact":"Significantly reduces CVD and diabetes risk."},
                     {"habit":"Monitor blood pressure","description":"Check and log blood pressure each morning.","target_risk":"hypertension","difficulty":"Easy","daily_time_minutes":5,"expected_impact":"Early detection of dangerous BP spikes."},
                     {"habit":"Reduce alcohol intake","description":"Limit to 1 drink per day maximum.","target_risk":"liver and cancer risk","difficulty":"Hard","daily_time_minutes":0,"expected_impact":"Reduces cancer and liver disease risk."}],
        "Very High":[{"habit":"Complete smoking cessation","description":"Join a free cessation program today.","target_risk":"cancer and respiratory disease","difficulty":"Hard","daily_time_minutes":15,"expected_impact":"Largest single modifiable risk reduction."},
                     {"habit":"Daily blood sugar check","description":"Check fasting glucose each morning.","target_risk":"diabetes","difficulty":"Medium","daily_time_minutes":5,"expected_impact":"Prevents undetected T2D progression."},
                     {"habit":"Strength training twice weekly","description":"Do 20-min bodyweight exercises 2x per week.","target_risk":"metabolic syndrome","difficulty":"Medium","daily_time_minutes":20,"expected_impact":"Improves insulin sensitivity."}],
    }
    habits = {}
    for _, row in patients.iterrows():
        pid = int(row["patient_id"])
        habits[pid] = {"patient_id":pid,"age":int(row["age"]),"risk_tier":row["risk_tier"],
                       "composite_risk":row["composite_risk"],
                       "habits":habit_templates.get(row["risk_tier"],habit_templates["Moderate"])}
    return patients, habits
