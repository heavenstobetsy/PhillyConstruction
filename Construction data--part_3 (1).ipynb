{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False\n",
      "Success - the MySageMakerInstance is in the us-east-1 region. You will use the 811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest container for your SageMaker endpoint.\n",
      "2020-01-24 15:18:21.824722\n",
      "S3 bucket created successfully\n",
      "imports complete\n"
     ]
    }
   ],
   "source": [
    "!python imports.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2020-01-24 15:19:43.247388\n"
     ]
    }
   ],
   "source": [
    "%run data_scraping.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Part 3: data cleaning and wrangling:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.a: Permit and Inspection data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "After choosing which elements to join on, there are ~80k addresses: 79922\n"
     ]
    }
   ],
   "source": [
    "#Merge permits and inspections:\n",
    "\n",
    "permit_inspect = permits.merge(inspect, on=['addresskey','censustract'],how='inner')\n",
    "\n",
    "print('After choosing which elements to join on, there are ~80k addresses: {}'.format(permit_inspect.addresskey.nunique()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "After choosing which elements to join on, there are ~69k addresses: 69883\n"
     ]
    }
   ],
   "source": [
    "# Filter Permits and Inspections data: so permitissuedate>=inspectioncompleted\n",
    "#I want to look at only data before the permits were issued, \n",
    "#as I'm predicting which permits need to be examined closer.\n",
    "\n",
    "permit_inspect=permit_inspect[permit_inspect['permitissuedate']>=permit_inspect['inspectioncompleted']]\n",
    "print('After choosing which elements to join on, there are ~69k addresses: {}'.format(permit_inspect.addresskey.nunique()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# #Drop data that I know is useless/has repeating features/inspectiontype feature is too messy:\n",
    "\n",
    "permit_inspect=permit_inspect.drop(['mostrecentinsp','descriptionofwork','council_district','permit_type_name','inspectiontype'], axis=1)\n",
    "#permit_inspect.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create dummies for inspectionstatus, and concat & group data:\n",
    "\n",
    "permit_inspect=pd.concat([permit_inspect, pd.get_dummies(permit_inspect.inspectionstatus)], 1).groupby([\"permitissuedate\",\"addresskey\",\"permitnumber\",\"permitdescription\",\"typeofwork\",\"censustract\",\"status\",\"applicantcapacity\",\"aptype\",\"casenumber\"]).sum().reset_index()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>permitissuedate</th>\n",
       "      <th>addresskey</th>\n",
       "      <th>permitnumber</th>\n",
       "      <th>permitdescription</th>\n",
       "      <th>typeofwork</th>\n",
       "      <th>censustract</th>\n",
       "      <th>status</th>\n",
       "      <th>applicantcapacity</th>\n",
       "      <th>aptype</th>\n",
       "      <th>Cancelled</th>\n",
       "      <th>Failed</th>\n",
       "      <th>Closed</th>\n",
       "      <th>HOLD</th>\n",
       "      <th>None</th>\n",
       "      <th>Passed</th>\n",
       "      <th>unique_insp_cases</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>21376</th>\n",
       "      <td>2007-11-05</td>\n",
       "      <td>593956</td>\n",
       "      <td>108629</td>\n",
       "      <td>ZONING/USE PERMIT</td>\n",
       "      <td>ADD</td>\n",
       "      <td>7.0</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>ATTORNEY</td>\n",
       "      <td>CD ENFORCE</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      permitissuedate  addresskey permitnumber  permitdescription typeofwork  \\\n",
       "21376      2007-11-05      593956       108629  ZONING/USE PERMIT        ADD   \n",
       "\n",
       "       censustract     status applicantcapacity      aptype  Cancelled  \\\n",
       "21376          7.0  COMPLETED          ATTORNEY  CD ENFORCE          0   \n",
       "\n",
       "       Failed  Closed  HOLD  None  Passed  unique_insp_cases  \n",
       "21376       0       0     0     0       1                 44  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Summarize inspection data by grouping:\n",
    "\n",
    "permit_inspect = (permit_inspect.groupby(['permitissuedate', 'addresskey', 'permitnumber', 'permitdescription', 'typeofwork', \n",
    "                                           'censustract', 'status', 'applicantcapacity','aptype',\n",
    "                                           'Cancelled','Failed','Closed','HOLD','None','Passed'])['casenumber']\n",
    "                   .nunique()\n",
    "                                           .reset_index().rename(columns={'casenumber':'unique_insp_cases'}))\n",
    "permit_inspect[(permit_inspect['addresskey']==593956)].head(1)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>permitissuedate</th>\n",
       "      <th>addresskey</th>\n",
       "      <th>permitnumber</th>\n",
       "      <th>permitdescription</th>\n",
       "      <th>typeofwork</th>\n",
       "      <th>censustract</th>\n",
       "      <th>status</th>\n",
       "      <th>applicantcapacity</th>\n",
       "      <th>aptype</th>\n",
       "      <th>Cancelled</th>\n",
       "      <th>Failed</th>\n",
       "      <th>Closed</th>\n",
       "      <th>HOLD</th>\n",
       "      <th>None</th>\n",
       "      <th>Passed</th>\n",
       "      <th>unique_insp_cases</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>14228</th>\n",
       "      <td>2007-11-05</td>\n",
       "      <td>593956</td>\n",
       "      <td>108629</td>\n",
       "      <td>ZONING/USE PERMIT</td>\n",
       "      <td>ADD</td>\n",
       "      <td>7.0</td>\n",
       "      <td>COMPLETED</td>\n",
       "      <td>ATTORNEY</td>\n",
       "      <td>CD ENFORCE</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      permitissuedate  addresskey permitnumber  permitdescription typeofwork  \\\n",
       "14228      2007-11-05      593956       108629  ZONING/USE PERMIT        ADD   \n",
       "\n",
       "       censustract     status applicantcapacity      aptype  Cancelled  \\\n",
       "14228          7.0  COMPLETED          ATTORNEY  CD ENFORCE          0   \n",
       "\n",
       "       Failed  Closed  HOLD  None  Passed  unique_insp_cases  \n",
       "14228       2       0     0     0       2                 47  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Reset index, and sum inspection data\n",
    "\n",
    "permit_inspect=(permit_inspect.groupby(['permitissuedate', 'addresskey', 'permitnumber', 'permitdescription', 'typeofwork', \n",
    "                                           'censustract', 'status', 'applicantcapacity','aptype']).sum().reset_index())\n",
    "permit_inspect[(permit_inspect['addresskey']==593956)].head(1)   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Count of unique values in each column:\n",
      "permitissuedate        3401\n",
      "addresskey            66299\n",
      "permitnumber         277734\n",
      "permitdescription        16\n",
      "typeofwork               72\n",
      "censustract             363\n",
      "status                    5\n",
      "applicantcapacity        27\n",
      "aptype                    8\n",
      "Cancelled                 7\n",
      "Failed                   66\n",
      "Closed                   35\n",
      "HOLD                      2\n",
      "None                      9\n",
      "Passed                   28\n",
      "unique_insp_cases       127\n",
      "dtype: int64\n",
      "Permit issue max date: 2020-01-23\n"
     ]
    }
   ],
   "source": [
    "#Sanity check the data--check out the unique column counts to look for discrepancies or possible side avenues \n",
    "#for analysis\n",
    "\n",
    "uniqueValues = permit_inspect.nunique()\n",
    " \n",
    "print('Count of unique values in each column:')\n",
    "print(uniqueValues)\n",
    "\n",
    "print('Permit issue max date: {}'.format(permit_inspect.permitissuedate.max()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.b: Property Violations data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/pandas/core/indexing.py:543: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy\n",
      "  self.obj[item] = s\n"
     ]
    }
   ],
   "source": [
    "#Get rid of status='ERROR', as that data is incorrect\n",
    "\n",
    "code_violations=code_violations[code_violations['status']!='ERROR']\n",
    "code_violations.loc[code_violations.status=='CMPLY', 'status'] = 'COMPLIED'\n",
    "code_violations.loc[code_violations.status=='CLOSED', 'status'] = 'RESOLVE'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3.b.1 Filter data, so permit issue date is after code violations:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# First step to filter out violations after permits, as I'm predicting which permits need to be examined closer.\n",
    "\n",
    "code_violations_dates=code_violations.filter(['addresskey','caseaddeddate','casenumber'], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data type cleanup\n",
    "\n",
    "permit_inspect['addresskey'] = permit_inspect['addresskey'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Temp dataset to filter on permitissuedate>=caseaddeddate:\n",
    "\n",
    "tmp = permit_inspect.merge(code_violations_dates, on=['addresskey'],how='left')\n",
    "tmp=tmp[(tmp['permitissuedate']>=tmp['caseaddeddate'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Addresskey count sanity check: 167677\n"
     ]
    }
   ],
   "source": [
    "#Filter on tmp dataset:\n",
    "\n",
    "code_violations_dates_filter_tmp=tmp.filter(['addresskey','caseaddeddate','casenumber'], axis=1)\n",
    "#print(code_violations_dates_filter_tmp.addresskey.count())\n",
    "\n",
    "#Drop duplicates to clean up dataset\n",
    "code_violations_dates_filter_tmp=code_violations_dates_filter_tmp.drop_duplicates()\n",
    "print('Addresskey count sanity check: {}'.format(code_violations_dates_filter_tmp.addresskey.count()))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Merge the corrected and filtered violations back on code_violations\n",
    "\n",
    "code_violations_dates_filter=code_violations_dates_filter_tmp.merge(code_violations, left_on=['casenumber','addresskey','caseaddeddate'],right_on=['casenumber','addresskey','caseaddeddate'],how='inner')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop any duplicates, and unneeded column(s)\n",
    "\n",
    "code_violations_dates_filter=code_violations_dates_filter.drop_duplicates()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data cleanup\n",
    "\n",
    "code_violations_dates_filter['casenumber'] = code_violations_dates_filter['casenumber'].astype(str)\n",
    "code_violations_dates_filter['casenumber int'] = code_violations_dates_filter['casenumber'].str.isnumeric().astype(int)\n",
    "\n",
    "#A 'casenumber int' column was added, as the casenumers feature had nonsense data--I'm checking to make sure there is \n",
    "#a valid casenumber. I then delete the 'casenumber int' column\n",
    "\n",
    "code_violations_dates_filter=code_violations_dates_filter[code_violations_dates_filter['casenumber int']==1]\n",
    "code_violations_dates_filter=code_violations_dates_filter.drop(['casenumber int'], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of unique unsafe violations: 3708\n"
     ]
    }
   ],
   "source": [
    "print(\"Number of unique unsafe violations: {}\".format(unsafe_violations.casenumber.nunique()))\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.c: Unsafe Property Violations data:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3.c.1 Filter data, so I'm not including the relevent unsafe violation when predicting unsafe status. Aka drop the most recent inspection that lead to unsafe status:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "unsafe_violations_tmp=unsafe_violations.filter(['addresskey','violationdate','casenumber'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Data clean up\n",
    "\n",
    "code_violations_dates_filter['casenumber'] = code_violations_dates_filter['casenumber'].astype(int)\n",
    "unsafe_violations_tmp['casenumber'] = unsafe_violations_tmp['casenumber'].astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Merge code_violations and unsafe_violations for filter\n",
    "\n",
    "code_violations_dates_filter_unsafe=code_violations_dates_filter.merge(unsafe_violations_tmp, left_on=['casenumber','addresskey'],right_on=['casenumber','addresskey'],how='inner')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Filter on tmp dataset to remove certain unsafe violations & drop any duplicates:\n",
    "\n",
    "code_violations_dates_filter_unsafe_remove_these=code_violations_dates_filter_unsafe.filter(['addresskey','casenumber','violationdate_y'], axis=1)\n",
    "code_violations_dates_filter_unsafe_remove_these=code_violations_dates_filter_unsafe_remove_these.drop_duplicates()\n",
    "code_violations_dates_filter_unsafe_remove_these.rename(columns={'casenumber': 'casenumber_unsafe','violationdate_y': 'violationdate_unsafe'}, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Merge the corrected and filtered violations back on code_violations\n",
    "\n",
    "code_violations_dates_prefilter=code_violations_dates_filter.merge(code_violations_dates_filter_unsafe_remove_these, left_on=['casenumber','addresskey'],right_on=['casenumber_unsafe','addresskey'],how='left')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Filter out casenumbers that are the same\n",
    "\n",
    "code_violations_dates_filtered=code_violations_dates_prefilter[code_violations_dates_prefilter['casenumber']!=code_violations_dates_prefilter['casenumber_unsafe']]\n",
    "#code_violations_dates_filtered.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total unique addresskeys: 58398\n",
      "Removed unique addresskeys: 374\n",
      "New filtered dataset unique addresskeys: 58320\n"
     ]
    }
   ],
   "source": [
    "print(\"Total unique addresskeys: {}\".format(code_violations_dates_filter.addresskey.nunique()))\n",
    "print(\"Removed unique addresskeys: {}\".format(code_violations_dates_filter_unsafe_remove_these.addresskey.nunique()))\n",
    "print(\"New filtered dataset unique addresskeys: {}\".format(code_violations_dates_filtered.addresskey.nunique()))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>addresskey</th>\n",
       "      <th>aptype</th>\n",
       "      <th>caseresolutioncode</th>\n",
       "      <th>violationdescription</th>\n",
       "      <th>status</th>\n",
       "      <th>casestatus</th>\n",
       "      <th>prioritydesc</th>\n",
       "      <th>code_violations_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>126852</th>\n",
       "      <td>356167</td>\n",
       "      <td>CD ENFORCE</td>\n",
       "      <td>CLOSE</td>\n",
       "      <td>VIOL NONE FOUND</td>\n",
       "      <td>COMPLIED</td>\n",
       "      <td>CLOSED</td>\n",
       "      <td>NON HAZARDOUS</td>\n",
       "      <td>55</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        addresskey      aptype caseresolutioncode violationdescription  \\\n",
       "126852      356167  CD ENFORCE              CLOSE      VIOL NONE FOUND   \n",
       "\n",
       "          status casestatus   prioritydesc  code_violations_count  \n",
       "126852  COMPLIED     CLOSED  NON HAZARDOUS                     55  "
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Group the data by violations type, address, status, and case resolution to see if repeating violations help \n",
    "#predict unsafe buildings\n",
    "\n",
    "code_violations_grouped=code_violations_dates_filtered.groupby([\"addresskey\",\"aptype\",\"caseresolutioncode\",\"violationdescription\",\"status\",\"casestatus\",\"prioritydesc\"]).size().reset_index(name='code_violations_count').sort_values(['code_violations_count'], ascending=False)\n",
    "code_violations_grouped.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.d.1 Flatten out code violations table, create dummy variables, and drop aptype and violationdescription (there are 1900 options!!) (Note: Ideally, I'd like to go back and group common violation descriptions by highly ranked keywords)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#Create dummy variable for feature='prioritydesc', concat, and summarize by grouping:\n",
    "\n",
    "code_violations_grouped=pd.concat([code_violations_grouped, pd.get_dummies(code_violations_grouped.prioritydesc)], 1).groupby(['addresskey','caseresolutioncode','violationdescription','status']).sum().reset_index()\n",
    "# code_violations_grouped[code_violations_grouped['addresskey']==1038]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create dummy variable for feature='status', concat, and summarize by grouping:\n",
    "\n",
    "code_violations_grouped=pd.concat([code_violations_grouped, pd.get_dummies(code_violations_grouped.status)], 1).groupby(['addresskey','caseresolutioncode','violationdescription']).sum().reset_index()\n",
    "# code_violations_grouped[code_violations_grouped['addresskey']==1038]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create dummy variable for feature='status', concat, and summarize by grouping:\n",
    "\n",
    "code_violations_grouped=pd.concat([code_violations_grouped, pd.get_dummies(code_violations_grouped.caseresolutioncode)], 1).groupby(['addresskey']).sum().reset_index()\n",
    "# code_violations_grouped[code_violations_grouped['addresskey']==1038]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.d.2 Join code_violations_grouped back on to permit_inspect to finally get a clean dataset:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = permit_inspect.merge(code_violations_grouped, on=['addresskey'],how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>permitissuedate</th>\n",
       "      <th>addresskey</th>\n",
       "      <th>permitnumber</th>\n",
       "      <th>permitdescription</th>\n",
       "      <th>typeofwork</th>\n",
       "      <th>censustract</th>\n",
       "      <th>status</th>\n",
       "      <th>applicantcapacity</th>\n",
       "      <th>Cancelled</th>\n",
       "      <th>Failed</th>\n",
       "      <th>...</th>\n",
       "      <th>RFU</th>\n",
       "      <th>SCA</th>\n",
       "      <th>SITE</th>\n",
       "      <th>SR</th>\n",
       "      <th>STP</th>\n",
       "      <th>UNF</th>\n",
       "      <th>UNFND</th>\n",
       "      <th>UNSUB</th>\n",
       "      <th>VIO</th>\n",
       "      <th>VOID</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2007-01-02</td>\n",
       "      <td>29921</td>\n",
       "      <td>46558</td>\n",
       "      <td>ALTERATION PERMIT</td>\n",
       "      <td>MAJOR</td>\n",
       "      <td>1.0</td>\n",
       "      <td>EXPIRED</td>\n",
       "      <td>CONTRACTOR</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 65 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  permitissuedate  addresskey permitnumber  permitdescription typeofwork  \\\n",
       "0      2007-01-02       29921        46558  ALTERATION PERMIT      MAJOR   \n",
       "\n",
       "   censustract   status applicantcapacity  Cancelled  Failed  ...  RFU  SCA  \\\n",
       "0          1.0  EXPIRED        CONTRACTOR          0       4  ...  0.0  0.0   \n",
       "\n",
       "   SITE   SR  STP  UNF  UNFND  UNSUB  VIO  VOID  \n",
       "0   0.0  0.0  0.0  0.0    0.0    0.0  0.0   0.0  \n",
       "\n",
       "[1 rows x 65 columns]"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Data clean up\n",
    "df.rename(columns={'status_x': 'permit_status','status_y': 'violation_status',}, inplace=True)\n",
    "df=df.drop(['aptype'], axis=1)\n",
    "df.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.d.3 Add in an unsafe violation flag:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "unsafe_violations_filter=unsafe_violations.filter(['addresskey','violationdate'], axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "unsafe_violations_filtered=df.merge(unsafe_violations_filter, left_on=['addresskey'],right_on=['addresskey'],how='left')\n",
    "unsafe_violations_filtered['unsafe_building']=0\n",
    "unsafe_violations_filtered.loc[unsafe_violations_filtered.violationdate >'2010-01-01', 'unsafe_building'] = 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "unsafe_violations_filtered[\"violationdate\"]= pd.to_datetime(unsafe_violations_filtered[\"violationdate\"]) \n",
    "unsafe_violations_filtered[\"permitissuedate\"]= pd.to_datetime(unsafe_violations_filtered[\"permitissuedate\"]) \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.d.4 Make sure the unsafe violation date is after permit date:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=unsafe_violations_filtered[(unsafe_violations_filtered['permitissuedate']<unsafe_violations_filtered['violationdate'])|(unsafe_violations_filtered.violationdate.isnull())]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Sanity check: drop any dupkicates\n",
    "\n",
    "df=df.drop_duplicates()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.d.5 Get rid of permit status, create and sum dummy variables, & group the number of inspection cases:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>addresskey</th>\n",
       "      <th>permitnumber</th>\n",
       "      <th>permitdescription</th>\n",
       "      <th>typeofwork</th>\n",
       "      <th>censustract</th>\n",
       "      <th>unsafe_building</th>\n",
       "      <th>code_violations_count</th>\n",
       "      <th>Cancelled</th>\n",
       "      <th>Failed</th>\n",
       "      <th>Closed</th>\n",
       "      <th>...</th>\n",
       "      <th>ONSITE CONTACT</th>\n",
       "      <th>OPERATOR OF BUSINESS</th>\n",
       "      <th>OWNER</th>\n",
       "      <th>OWNER CONTACT</th>\n",
       "      <th>PROF</th>\n",
       "      <th>PROPERTY MANAGER</th>\n",
       "      <th>RESPONSIBLE COMPANY CONTACT</th>\n",
       "      <th>SAFETY OFFICER</th>\n",
       "      <th>SPECIAL INSPECTOR</th>\n",
       "      <th>TENANT</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1038</td>\n",
       "      <td>825449</td>\n",
       "      <td>ELECTRICAL PERMIT</td>\n",
       "      <td>EZELEC</td>\n",
       "      <td>176.0</td>\n",
       "      <td>0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 90 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   addresskey permitnumber  permitdescription typeofwork  censustract  \\\n",
       "0        1038       825449  ELECTRICAL PERMIT     EZELEC        176.0   \n",
       "\n",
       "   unsafe_building  code_violations_count  Cancelled  Failed  Closed  ...  \\\n",
       "0                0                    6.0          0       4       0  ...   \n",
       "\n",
       "   ONSITE CONTACT  OPERATOR OF BUSINESS  OWNER  OWNER CONTACT  PROF  \\\n",
       "0               0                     0      0              0     0   \n",
       "\n",
       "   PROPERTY MANAGER  RESPONSIBLE COMPANY CONTACT  SAFETY OFFICER  \\\n",
       "0                 0                            0               0   \n",
       "\n",
       "   SPECIAL INSPECTOR  TENANT  \n",
       "0                  0       0  \n",
       "\n",
       "[1 rows x 90 columns]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Create dummy variable for feature='applicantcapacity', concat, and summarize by grouping:\n",
    "\n",
    "df2=pd.concat([df, pd.get_dummies(df.applicantcapacity)], 1).groupby(['addresskey','permitnumber','permitdescription','typeofwork','censustract','unsafe_building','code_violations_count']).sum().reset_index()\n",
    "df2.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create dummy variable for feature='typeofwork', concat, and summarize by grouping:\n",
    "\n",
    "df3=pd.concat([df2, pd.get_dummies(df2.typeofwork)], 1).groupby(['addresskey','permitnumber','permitdescription','censustract','unsafe_building','code_violations_count']).sum().reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create dummy variable for feature='permitdescription', concat, and summarize by grouping:\n",
    "\n",
    "df3=pd.concat([df3, pd.get_dummies(df3.permitdescription)], 1).groupby(['addresskey','permitnumber','censustract','unsafe_building','code_violations_count']).sum().reset_index()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Sanity check, drop any duplicates\n",
    "df3=df3.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill N/A with 0, rename the dataset and columns\n",
    "\n",
    "df3.fillna(0, inplace=True)\n",
    "permit_inspect_unsafe=df3\n",
    "permit_inspect_unsafe.rename(columns={'CLOSED': 'CLOSED_1'}, inplace=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_python3",
   "language": "python",
   "name": "conda_python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
