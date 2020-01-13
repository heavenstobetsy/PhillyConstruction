import pandas as pd   

## Data sources:

#Permits data API link
permits = pd.read_csv("https://phl.carto.com/api/v2/sql?q=SELECT+addresskey,+censustract,+permitnumber,+permitdescription,+permit_type_name,+typeofwork,+permitissuedate,+status,+applicantcapacity,+mostrecentinsp,+council_district,+descriptionofwork+FROM+li_permits&filename=li_permits&format=csv&skipfields=cartodb_id",low_memory=False)


#Inspection data API link
inspect=pd.read_csv("https://phl.carto.com/api/v2/sql?q=SELECT+addresskey,+censustract,+aptype,+inspectiontype,+inspectioncompleted,+inspectionstatus,+casenumber+FROM+li_case_inspections&filename=li_case_inspections&format=csv&skipfields=cartodb_id",low_memory=False)


# #Code violations data API link
code_violations=pd.read_csv("https://phl.carto.com/api/v2/sql?q=SELECT+addresskey,+casenumber,+aptype,+caseaddeddate,+caseresolutioncode,+violationdate,+violationdescription,+status,+casestatus,+casegroup,+casepriority,+prioritydesc,+caseresolutiondate+FROM+li_violations&filename=li_violations&format=csv&skipfields=cartodb_id",low_memory=False)


#Unsafe violations data API link
unsafe_violations=pd.read_csv("https://phl.carto.com/api/v2/sql?q=SELECT+addresskey,+caseresolutiondate,+caseresolutioncode,+violationdate,+casegrp,+violationdescription,+mostrecentinsp,+casenumber+FROM+li_unsafe&filename=li_unsafe&format=csv&skipfields=cartodb_id")


print(pd.datetime.now())
