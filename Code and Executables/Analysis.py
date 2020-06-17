import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joypy

df= pd.DataFrame(pd.read_csv('articles_data.csv'))

#print(df.head())

#no. of articles in each topic(bar chart) DONE
#topic wise count for each source (bar chart) DONE
#no. of reliable and unreliable articles in each source(grouped bar chart)
#caa reliable from all sources and other topic(bar chart)
#reliability score
#Pie chart for scores on few articles
#Heatmap for reliability scores.

# def articles_in_topics():
#     articles= df['Title']
    
#     topics = df['Topic_Name']
       
#     plt.bar(topics,range(len(articles)), align='center',color='blue',alpha=0.5)
#     plt.xticks(rotation=90)
    
#     plt.show()
#articles_in_topics()
 
# def caa_source():
#     plt.figure(figsize=(14,10), dpi= 80)
#     #the news topic CAA in each source
#     caa_topic= df.loc[df.Topic_Name =='caa-nrc','Topic_Name']
#     source_caa = df.loc[df.Topic_Name =='caa-nrc','Source']
#     caa=plt.bar(source_caa,range(len(caa_topic)),align='center',color='red',alpha=0.5)
#     plt.xticks(rotation=45)
#     plt.ylabel('CAA-NRC')
#     plt.show()

#caa_source()

# def corona_source():
#     #the news topic corona virus in each source
#     corona_topic = df.loc[df.Topic_Name == 'coronavirus' , 'Topic_Name']
#     source_corona = df.loc[df.Topic_Name =='coronavirus','Source']
#     corona=plt.bar(source_corona,range(len(corona_topic)),align='center',color='blue',alpha=0.5)
#     plt.xticks(rotation=90)
#     plt.ylabel('CORONA VIRUS')
#     plt.show()

#corona_source()

# def reliable_sources():
#     #Shows the articles of both topics being reliable 
#     reliable = df.loc[df.Reliability_Label == 'Reliable' , 'Source']    
#     plt.bar(range(len(reliable)), reliable,align='center',color='green',alpha=0.5 )
#     plt.show()

#reliable_sources()

# def unreliable_sources():
#     #Shows the articles of both topics being unreliable 
#     unreliable = df.loc[df.Reliability_Label == 'Not Reliable' , 'Source']    
#     plt.bar(range(len(unreliable)), unreliable,align='center',color='black')
#     plt.show()
#unreliable_sources()

def caa_reliable_unreliable():
    plt.figure(figsize=(10,10), dpi= 80)
    #Shows reliable CAA articles in the sources
    caa= df.loc[df.Topic_Name =='caa-nrc']
    caa_reliable = caa[caa.Reliability_Label == 'Reliable']
    # print(caa_reliable)
    caa_unreliable = caa[caa.Reliability_Label == 'Not Reliable']
   # print(source_reliable)
   
    # source_unreliable = caa.loc[caa.Reliability_Label == 'Not Reliable', 'Source']
    # print(cov_reliable)
    # plt.bar(source_reliable,range(len(caa_reliable)),align='center',color='blue',alpha=0.5)
    # plt.xticks(rotation=90)
    # plt.show()
   
    count1 = caa_reliable.groupby('Source').count()[['Reliability_Label']]
    count2 = caa.groupby('Source').count()[['Reliability_Label']]
    l1= list(count1['Reliability_Label'])
    l2 = list(count2['Reliability_Label'])
    l3=[]
    for i in range(len(l1)):
        l3.append(l2[i]-l1[i])



    print(len(l1))
    print(len(l3))
    #print(count)
    #print(l.value_counts())
    # width=0.1
    plt.barh(l1,range(len(l1)) , tick_label = list(set(df['Source'])))
    plt.xticks(rotation=90)
    plt.title("Stacked Bar Chart - Number of reliable and unreliable articles related to CAA-NRC")
    plt.barh(l3,range(len(l3)))
    plt.xlabel('Source')
    # plt.invert_yaxis()
    plt.show()

caa_reliable_unreliable()

def cov_reliable_unreliable():
    plt.figure(figsize=(10,10), dpi= 40)
    #Shows reliable CAA articles in the sources
    cov= df.loc[df.Topic_Name =='coronavirus']
    cov_reliable = cov[cov.Reliability_Label == 'Reliable']
    # print(caa_reliable)
    cov_unreliable = cov[cov.Reliability_Label == 'Not Reliable']
   # print(source_reliable)
   
    # source_unreliable = caa.loc[caa.Reliability_Label == 'Not Reliable', 'Source']
    # print(cov_reliable)
    # plt.bar(source_reliable,range(len(caa_reliable)),align='center',color='blue',alpha=0.5)
    # plt.xticks(rotation=90)
    # plt.show()
   
    count1 = cov_reliable.groupby('Source').count()[['Reliability_Label']]
    count2 = cov.groupby('Source').count()[['Reliability_Label']]
    l1= list(count1['Reliability_Label'])
    l2 = list(count2['Reliability_Label'])
    l3=[]
    for i in range(len(l1)):
        l3.append(l2[i]-l1[i])



    print(len(l1))
    print(len(l3))
    #print(count)
    #print(l.value_counts())
    # width=0.1
    plt.bar(range(len(l1)), l1, tick_label = list(set(df['Source'])))
    plt.xticks(rotation=90)
    plt.title("Stacked Bar Chart - Number of reliable and unreliable articles related to Corona Virus")

    plt.bar(range(len(l3)), l3, bottom=l1)
    plt.xlabel('Source')
    plt.show()
#cov_reliable_unreliable()

# def cov_unreliable():
#     #Shows unreliable Corona Virus articles in the sources
#     cov= df.loc[df.Topic_Name =='coronavirus']
#     cov_unreliable = cov.loc[cov.Reliability_Label == 'Not Reliable', 'Reliability_Label']
#     # print(caa_reliable)
#     source_unreliable = cov.loc[cov.Reliability_Label == 'Not Reliable', 'Source']
#     # print(cov_reliable)
#     plt.bar(source_unreliable,range(len(cov_unreliable)),align='center',color='blue',alpha=0.5)
#     plt.xticks(rotation=90)
#     plt.show()
# cov_unreliable()


# def pie():
    
#     #source = pd.DataFrame(source)
#     # source = source.iloc[:4]
#     df['freq'] = df.groupby('Source')['Source'].transform('count')
#     source= df.loc[df.Topic_Name =='caa-nrc']
#     #print(source)
#     freq = source[['freq']]
    
#     label= source[['Source Score','Strong Words Score','Neutrality Score','Reliability_Score']]
#     print(label)
#     #rel=data[['Neutrality Score','Source Score']]
#     colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
#     explode = (0.1, 0, 0, 0, 0) 
#     plt.pie(freq, labels= label,  colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
#     plt.show()

# pie()

def density_reliable():
    # score = df['Reliability_Score']
    # source = df['Source']
    # topic = len(df['Topic_Name'])
    # data = pd.DataFrame({
    #                'Source': source,
    #                'Reliability': score,
    #                'Topic':topic
    #               })
    # data_heatmap = data.pivot_table('Source','Topic','Reliability',aggfunc=np.mean)
    # sns.heatmap(data_heatmap)
    # plt.show()
    
    # sns.kdeplot(df.loc[df['Reliability_Label'] == 'Reliable', "Reliability_Score"], shade=True, color="g", label="Reliable articles score", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Reliable', "Neutrality Score"], shade=True, color="deeppink", label="Neutrality score for reliable articles", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Reliable', "Source Score"], shade=True, color="dodgerblue", label="Source score for reliable articles", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Reliable', "Strong Words Score"], shade=True, color="orange", label="Strong Words score for reliable articles", alpha=.7)
    

    # Decoration
    plt.title('Density plot of scores of reliable articles', fontsize=22)
    plt.legend()
    plt.show()
#density_reliable()

def density_unreliable():
    # plt.figure(figsize=(16,10), dpi= 80)
    # sns.kdeplot(df.loc[df['Reliability_Label'] == 'Not Reliable', "Reliability_Score"], shade=True, color="g", label="Unreliable articles score", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Not Reliable', "Neutrality Score"], shade=True, color="deeppink", label="Neutrality score for unreliable articles", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Not Reliable', "Source Score"], shade=True, color="dodgerblue", label="Source score for unreliable articles", alpha=.7)
    sns.kdeplot(df.loc[df['Reliability_Label'] == 'Not Reliable', "Strong Words Score"], shade=True, color="orange", label="Strong Words score for unreliable articles", alpha=.7)
    plt.title('Density plot of scores of unreliable articles', fontsize=22)
    plt.legend()
    plt.show()

density_unreliable()

# def dist():
#     plt.figure(figsize=(13,10), dpi= 80)
#     sns.distplot(df.loc[df['Source'] == 'The Times of India', "Reliability_Score"], color="dodgerblue", label="The Times of India", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
#     sns.distplot(df.loc[df['Source'] == 'DNA', "Reliability_Score"], color="orange", label="DNA", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
#     #sns.distplot(df.loc[df['Source'] == 'minivan', "cty"], color="g", label="minivan", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
#     # plt.ylim(0, 0.35)

#     # Decoration
#     plt.title('Density Plot of Reliability score by TOI and DNA', fontsize=22)
#     plt.legend()
#     plt.show()

# dist()

# def scatter():
#     sns.scatterplot(x=df['Source Score'], y=df['Reliability_Score'], data=df)
#     plt.show()

# scatter()
def joy():

    fig, axes = joypy.joyplot(df, by="Source", column=["Neutrality Score","Source Score","Strong Words Score"],figsize=(8,8),legend=True)
    # plt.title("Joy Plot showing Neutrality score, Source score, Strong words score for Sources")
    plt.show()
# joy()



# reliable()