#!/usr/bin/perl
########################################################################################################################
#  Creater        :sunzhe3
#  Creation Time  :20170917
#  Description    :用户画像性别预测标签姓名特征加工
#  Modify By      :
#  Modify Time    :
#  Modify Content :
#  Script Version :1.0.3
########################################################################################################################
use strict;
use jrjtcommon;
use un_pswd;
use Common::Hive;
use zjcommon;

##############################################
#默认STINGER运行，失败后HIVE运行，可更改Runner和Retry_Runner
#修改最终生成表库名和表名
##############################################

my $Runner = "STINGER";
my $Retry_Runner = "HIVE";
my $DB = "";
my $TABLE = "";
##############################################

if ( $#ARGV < 0 ) { exit(1); }
my $CONTROL_FILE = $ARGV[0];
my $SYS = substr(${CONTROL_FILE}, 0, 3);
my $JOB = substr(${CONTROL_FILE}, 4, length(${CONTROL_FILE})-17);

#当日 yyyy-mm-dd
my $TX_DATE = substr(${CONTROL_FILE},length(${CONTROL_FILE})-12, 4).'-'.substr(${CONTROL_FILE},length(${CONTROL_FILE})-8, 2).'-'.substr(${CONTROL_FILE},length(${CONTROL_FILE})-6, 2);

my $TXDATE = substr($TX_DATE, 0, 4).substr($TX_DATE, 5, 2).substr($TX_DATE, 8, 2);                        #当日 yyyymmdd
my $TX_MONTH = substr($TX_DATE, 0, 4).'-'.substr($TX_DATE, 5, 2);                                          #当日所在月 yyyy-mm
my $TXMONTH = substr($TX_DATE, 0, 4).substr($TX_DATE, 5, 2);                                               #当日所在月 yyyymm
my $TX_PREV_DATE = getPreviousDate($TX_DATE);                                                               #前一天 yyyy-mm-dd
my $TX_NEXT_DATE = getNextDate($TX_DATE);                                                                   #下一天 yyyy-mm-dd
my $TXPDATE = substr(${TX_PREV_DATE},0,4).substr(${TX_PREV_DATE},5,2).substr(${TX_PREV_DATE},8,2);        #前一天 yyyymmdd
my $TXNDATE = substr(${TX_NEXT_DATE},0,4).substr(${TX_NEXT_DATE},5,2).substr(${TX_NEXT_DATE},8,2);        #下一天 yyyymmdd
my $CURRENT_TIME = getNowTime();
my $TX_YEAR = substr($TX_DATE, 0, 4);#当年 yyyy

########################################################################################################################
# Write SQL For Your APP
sub getsql
{
    my @SQL_BUFF=();
    #########################################################################################
    ####################################以下为SQL编辑区######################################
    #########################################################################################
$SQL_BUFF[0]=qq(
set mapreduce.job.name=dmt_upf_gender_prediction_name_feature_s_d_0;
use dmt;
CREATE TABLE IF NOT EXISTS dmt.dmt_upf_gender_prediction_name_feature_s_d(
jd_pin  string             comment  'jd_pin'
,name   string             comment  '姓名加密'
,name_d2  string             comment  '姓名倒数第二个字加密'
,name_d1  string             comment  '姓名倒数第一个字加密'
,name_d21  string             comment  '姓名最后两个字组合加密'
,gender  string   comment  '性别男1女0'
)COMMENT '用户画像性别预测标签姓名特征加工'
PARTITIONED BY (dt string  comment '日期分区')
STORED AS ORC;
);


$SQL_BUFF[1]=qq(
set mapreduce.job.name=dmt_upf_gender_prediction_name_feature_s_d_1;
use dmt;
insert overwrite table dmt.dmt_upf_gender_prediction_name_feature_s_d partition(dt='$TX_DATE')
select jd_pin
,name
,name_d2
,name_d1
,name_d21 
,case when length(credentialsno_unpass)=15 and cast(substr(credentialsno_unpass,-1,1) as bigint)%2=1 then '1'
             when length(credentialsno_unpass)=15 and cast(substr(credentialsno_unpass,-1,1) as bigint)%2=0 then '0'
             when length(credentialsno_unpass)=18 and cast(substr(credentialsno_unpass,-2,1) as bigint)%2=1 then '1'
             when length(credentialsno_unpass)=18 and cast(substr(credentialsno_unpass,-2,1) as bigint)%2=0 then '0'
        else NULL end as gender          --性别
from 
(
select pin as jd_pin
,name
,password(substr(trim(unpassword(name, '$PWD::password')),-2,1), '$PWD::password') as name_d2
,password(substr(trim(unpassword(name, '$PWD::password')),-1,1), '$PWD::password') as name_d1
,password(substr(trim(unpassword(name, '$PWD::password')),-2,2), '$PWD::password') as name_d21
,trim(unpassword(credentialsno, '$PWD::password')) as credentialsno_unpass
,row_number() over (partition by pin order by updatetime desc) as rn  --有少量重复
from  dwb.DWB_MEM_AUTH_USER_PRM_S_D 
where dt = '$TX_DATE'
and authtype not in ('4','5','6','19','20')  --去掉弱实名
and length(trim(unpassword(name, '$PWD::password'))) > 0 
and length(trim(unpassword(credentialsno, '$PWD::password'))) > 0
and pin is not null
) t
where rn =1
and case when length(credentialsno_unpass)=15 and cast(substr(credentialsno_unpass,-1,1) as bigint)%2=1 then '1'
             when length(credentialsno_unpass)=15 and cast(substr(credentialsno_unpass,-1,1) as bigint)%2=0 then '0'
             when length(credentialsno_unpass)=18 and cast(substr(credentialsno_unpass,-2,1) as bigint)%2=1 then '1'
             when length(credentialsno_unpass)=18 and cast(substr(credentialsno_unpass,-2,1) as bigint)%2=0 then '0'
        else null  end  is not null
;
);


$SQL_BUFF[2]=qq(
set mapreduce.job.name=dmt_upf_gender_prediction_name_feature_s_d_2;
use dmt;
ALTER TABLE dmt.dmt_upf_gender_prediction_name_feature_s_d DROP IF EXISTS PARTITION(dt < '$TX_DATE');
);


    #############################################################################################
    ########################################以上为SQL编辑区######################################
    #############################################################################################

    return @SQL_BUFF;
}

########################################################################################################################

sub main
{
    my $ret;

    my @sql_buff = getsql();

    for (my $i = 0; $i <= $#sql_buff; $i++) {
        $ret = Common::Hive::run_hive_sql($sql_buff[$i], ${Runner}, ${Retry_Runner});

        if ($ret != 0) {
            print getCurrentDateTime("SQL_BUFF[$i] Execute Failed");
            return $ret;
        }
        else {
            print getCurrentDateTime("SQL_BUFF[$i] Execute Success");
        }
    }

    return $ret;
}

########################################################################################################################
# program section
# To see if there is one parameter,
print getCurrentDateTime(" Startup Success ..");
print "SYS          : $SYS\n";
print "JOB          : $JOB\n";
print "TX_DATE      : $TX_DATE\n";
print "TXDATE       : $TXDATE\n";
print "Target TABLE : $TABLE\n";

my $rc = main();
if ( $rc != 0 ) {
    print getCurrentDateTime("Task Execution Failed"),"\n";
} else{
    print getCurrentDateTime("Task Execution Success"),"\n";
}
exit($rc);

