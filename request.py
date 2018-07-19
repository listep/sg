#encoding:utf-8
import requests
import json
count_all = 0
hit = 0
fout = open('request.out_ik_balance','w')

dict_query_all = dict()
dict_recall_hit = dict()
dict_predict_all = dict()

jiexi_count = 0

fei_chui_lei = ['qa','baike','chat']

count_vertical = 0
count_non_vertical = 0
pre_count_vertical = 0
pre_count_non_vertical = 0

vertical_hit = 0
non_vertical_hit = 0

# string = 'http://10.142.98.128:8080/search?query=今天这个歌好听吗'
#
# github_url = string
# data = {}
# r = requests.post(github_url, data=data)
#
# ans = json.loads(r.text)
# print (ans['answer'])



with open('domain_test_no_o2o_0620.tsv','r') as f:

    for line in f.readlines():
        count_all += 1
        l = line.strip().split('\t')
        line_query,line_domain = l[0],l[1]

        if line_domain not in fei_chui_lei:
            count_vertical += 1
        else:
            count_non_vertical += 1


        ##测试集的domain的统计

        if line_domain not in dict_query_all:
            dict_query_all[line_domain] = 1
        else:
            dict_query_all[line_domain] += 1

        string = 'http://10.142.98.128:8080/search?query='+ str(line_query)

        # string = 'http://10.142.98.128:8080/search?query=今天这个歌好听吗'

        github_url = string
        data = {}
        r = requests.post(github_url,data = data)

        ans = json.loads(r.text)
        # print ans['answer']

        # exit()

        ans = ans['answer']

        my_dict = dict()

        try:
            for query in ans:
                #print (query['fieldsJson'])#.encode('utf-8')

            #    print( query['fieldsJson'].split('\"')[7] )
            #     print(query)

                domain = query['fieldsJson'].split('\"')[7]

                if domain  not in my_dict:

                    my_dict[ domain] = 1

                else:
                    my_dict[ domain]  += 1
            jiexi_count += 1


            #print (ans)

            #print (my_dict)

            # if sorted(my_dict.items(),key= lambda a:a[1],reverse = True)[0][0]

            # my_dict['chat'] = 5
            # my_dict['play_command'] = 2

            ##@@  find the one appear most times and appear in the first if appear times are same~

            dict_sorted = sorted(my_dict.items(), key=lambda a: a[1], reverse=True)

            if len(dict_sorted)>=2:

                if dict_sorted[0][1] != dict_sorted[1][1]:

                    predict_domain = sorted(my_dict.items(), key=lambda a: a[1], reverse=True)[0][0]
                else:
                    to_be_see = []
                    for cell in dict_sorted:
                        if cell[1] == dict_sorted[0][1]:
                            to_be_see.append(cell[0])

                    for query in ans:

                        domain = query['fieldsJson'].split('\"')[7]

                        if domain in to_be_see:
                            predict_domain = domain
                            break
            else:
                predict_domain = dict_sorted[0][0]

            # print(dict_sorted)
            #
            # print(predict_domain)

            #print (predict_domain,line_domain)
        ##@@ predict correct!

            ##预测的domain的统计

            if predict_domain not in dict_predict_all:
                dict_predict_all[predict_domain] = 1
            else:
                dict_predict_all[predict_domain] += 1


            if predict_domain not in fei_chui_lei:
                pre_count_vertical += 1
            else:
                pre_count_non_vertical += 1

            ### 预测正确的统计

            if predict_domain == line_domain:
                hit += 1
                if predict_domain not in dict_recall_hit:
                    dict_recall_hit[predict_domain] = 1
                else:
                    dict_recall_hit[predict_domain] += 1

            ##同时在非垂类
            if predict_domain in fei_chui_lei and line_domain in fei_chui_lei :
                non_vertical_hit += 1
            ##同时不在非垂类，就是同时在垂类
            if predict_domain not in fei_chui_lei and line_domain not  in fei_chui_lei:
                vertical_hit += 1

        except:

            print(ans)
            print(str(line_query))
            print('exception')



        # break

print ('28 domain:',hit/count_all)

fout.write('28 domain correct/all_query:' + '\t'+str(hit/count_all) + '\n')

for key in dict_query_all:

    P = dict_recall_hit[key]/dict_predict_all[key]
    R = dict_recall_hit[key]/dict_query_all[key]
    F1 = 2*P*R/(P+R)
    count = dict_query_all[key]
    print('domain: ',key ,'count: ',count,'recall: ',R ,'precision: ',P,'F1: ',F1)
    # print()
    # fout.write('domain_precision:'  + '\t'+ str(key)  +'\t'+ str (dict_recall_hit[key]/dict_predict_all[key])+'\n' )

    fout.write( '\t'.join( ['domain: ' , str(key) ,'count: ',str(count),'recall: ',str(R ) ,'precision: ' , str (P),'F1: ',str(F1) ]) +'\n' )


# for key in dict_predict_all:
v_R = vertical_hit/count_vertical
v_P = vertical_hit/pre_count_vertical

v_F1 = 2*v_P*v_R/(v_P+v_R)

non_v_R = non_vertical_hit/count_non_vertical
non_v_P = non_vertical_hit/pre_count_non_vertical

non_F1 = 2*non_v_P*non_v_R/(non_v_P+non_v_R)

print('vertical:' ,'recall:',v_R,'precision:',v_P)
print('non_vertical:' ,'recall:',non_v_R,'precision:',non_v_P)

fout.write('\t'.join(['vertical:' ,'recall:', str(v_R),'precision:',str(v_P)  ,'v_F1',str(v_F1)  ]  )+'\n' )
fout.write('\t'.join(['non_vertical:' ,'recall:',str(non_v_R),'precision:',str(non_v_P)  ,'non_v_F1',str(non_F1) ] )+'\n')


vertical_acc = (non_vertical_hit+vertical_hit)/(count_vertical+count_non_vertical)

fout.write( '\t'.join(['non_vertical_hit,vertical_hit,count_vertical,count_non_vertical',str(non_vertical_hit),str(vertical_hit),str(count_vertical),str(count_non_vertical)]) +'\n' )
fout.write('jiexi_count : '+'\t' +str(jiexi_count) +'\n')
fout.write( '\t'.join(['vertical_acc',str(vertical_acc)]) + '\n')