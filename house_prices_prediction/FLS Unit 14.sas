ODS HTML CLOSE; ODS HTML;
dm 'log;clear;';
PROC IMPORT OUT= WORK.bodyfat
        DATAFILE= "C:\Users\dnguy\Desktop\3 Statistical Foundations\Unit 14 & 15 Project\bodyfatbig.csv" 
        DBMS=CSV REPLACE;
    GETNAMES=YES;
    DATAROW=2; 
RUN;

  data body2;
  set bodyfat;
  if _n_=39 then delete;
  if _n_=42 then delete;
  run;

/*Forward*/
proc glmselect data=body2;
model Fat=Age Height Neck Chest Abs Hip Thigh Knee Ankle Biceps ForeArms Wrist
/ selection=Forward(stop=CV) cvmethod=random(5) stats=adjrsq;
run;

/*Backward*/
proc glmselect data=body2;
model Fat=Age Height Neck Chest Abs Hip Thigh Knee Ankle Biceps ForeArms Wrist
/ selection=Backward(stop=CV) cvmethod=random(5) stats=adjrsq;
run;

/*Stepwise*/
proc glmselect data=body2;
model Fat=Age Height Neck Chest Abs Hip Thigh Knee Ankle Biceps ForeArms Wrist
/ selection=Stepwise(stop=CV) cvmethod=random(5) stats=adjrsq;
run;

