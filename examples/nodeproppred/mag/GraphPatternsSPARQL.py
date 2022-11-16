import  pandas as pd
import datetime
import requests
import traceback
import sys
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_offset', dest='start_offset', type=int, help='Add start_offset')
    args = parser.parse_args()
    start_offset=args.start_offset
    print("start_offset=",start_offset)
        
    # """./isql 127.0.0.1:1111 dba dba exec="set blobs on; sparql define output:format '"TSV"' select  ?s as ?subject ?p as ?predicate ?o as ?object from <http://mag.org>  where {?s <https://makg.org/has_venue> ?v. ?s ?p ?o.};"   > KGTOSA_MAG_StarQuery.tsv"""
    dic_mag_queries={
         "RS": """ select  ?s as ?subject ?p as ?predicate ?o as ?object  
                    where
                    {
                            select ?s ?p ?o
                            from <http://mag.org>
                            where
                            {
                                ?s <https://makg.org/has_venue> ?v.
                                ?s ?p ?o.
                                filter(!isBlank(?s))
                            }
                    }
                    """,
         "LS": """ select  ?s as ?subject ?p as ?predicate ?o as ?object  
                    where
                    {
                            select ?s2 as ?s ?p as ?p ?s as ?o
                            from <http://mag.org>
                            where
                            {
                                ?s <https://makg.org/has_venue> ?v.
                                ?s2 ?p ?s.
                                filter(!isBlank(?s2))
                            }
                    }
                    """,       
        "RRS": """select  ?s as ?subject ?p as ?predicate ?o as ?object 
                    where
                    {
                    select   ?o as ?s  ?p2 as ?p ?s2 as ?o
                    from <http://mag.org>
                    where
                    {               
                     ?o ?p2 ?s2.
                      {
                       select  distinct ?o
                       from <http://mag.org>
                       where                                     
                       {
                         ?s <https://makg.org/has_venue> ?v.
                         ?s ?p ?o.
                       }
                      }
                      filter(!isBlank(?o))
                     }
                    }
                    """,
        "LLS": """select  ?s as ?subject ?p as ?predicate ?o as ?object 
                    where
                    {
                    select   ?s2 as ?s  ?p2 as ?p ?o as ?o
                    from <http://mag.org>
                    where
                    {               
                     ?s2 ?p2 ?o.
                      {
                       select  distinct ?o
                       from <http://mag.org> 
                       where                                     
                       {
                         ?s <https://makg.org/has_venue> ?v.
                         ?o ?p ?s.
                       }
                      }
                      filter(!isBlank(?s2))
                     }
                    }
                    """,
         "RRRS": """select  ?s as ?subject ?p as ?predicate ?o as ?object 
                    where
                    {
                    select   ?o2 as ?s  ?32 as ?p ?o3 as ?o
                    from <http://mag.org>
                    where
                    {
                        ?o2 ?p3 ?o3.
                        {
                            select  distinct ?o2
                            from <http://mag.org>
                            where
                            {
                             ?o ?p2 ?o2.
                              {
                               select  distinct ?o
                               from <http://mag.org>
                               where                                     
                               {
                                 ?s <https://makg.org/has_venue> ?v.
                                 ?s ?p ?o.
                               }
                              }
                              filter(!isBlank(?o)).
                             }
                          }                           
                          filter(!isBlank(?o2)).
                     }
                    }
                    """,
        "LLLS": """select  ?s as ?subject ?p as ?predicate ?o as ?object 
                    where
                    {
                    select   ?s3 as ?s  ?p3 as ?p ?o2 as ?o
                    from <http://mag.org>
                    where
                    {  
                    ?s3 ?p3 ?o2.
                     {
                       select  distinct ?o2
                       from <http://mag.org> 
                       where                                     
                       {                       
                         ?o2 ?p2 ?o.
                          {
                           select  distinct ?o
                           from <http://mag.org> 
                           where                                     
                           {
                             ?s <https://makg.org/has_venue> ?v.
                             ?o ?p ?s.
                           }
                          }
                          filter(!isBlank(?o2)).
                       }
                      }
                     filter(!isBlank(?s3)).
                     }
                    }
                    """,
       
    }
    url = 'http://127.0.0.1:8890/sparql/'
    for key in dic_mag_queries.keys():
        dataset="KGTOSA_MAG_Paper_Venue_"+key
        query=dic_mag_queries[key]
        print("query=",query)
        print("usecase=",dataset)
        start_t = datetime.datetime.now()
        # query_rows_count=query.replace("?s as ?subject ?p as ?predicate ?o as ?object","count(*) as ?rows_count")
        # # query_rows_count+="\n limit 1"
        # body = {'query': query_rows_count }
        # headers = {'Content-Type': 'application/x-www-form-urlencoded',
        #            'Accept': 'text/tab-separated-values; charset=UTF-8'}
        # r = requests.post(url, data=body, headers=headers)
        # rows_count=int(r.text.replace("rows_count", "").replace("\n","").replace("\"",""))
        # # rows_count=dic_mag_queries_counts[key]
        rows_count=1731323666
        # rows_count=1027288463
        print("rows_count=",rows_count)
        q_start_t = datetime.datetime.now()
        batch_size = 100000
        batches_count=int(rows_count/batch_size)+1
        print("batches_count=",batches_count)
        with open("/shared_mnt/KGTOSA_MAG/"+dataset+'_offset_'+str(start_offset)+'.tsv', 'w') as f:
            q_start_t = datetime.datetime.now()
            # for idx, offset in enumerate(range(0, 1000000, batch_size)):
            for idx, offset in enumerate(range(start_offset, rows_count, batch_size)):
                try:
                    start_t = datetime.datetime.now()
                    body = {'query': query + "\n offset " + str(offset) + " limit " + str(batch_size)}
                    headers = {'Content-Type': 'application/x-www-form-urlencoded',
                               'Accept': 'text/tab-separated-values; charset=UTF-8'}
                    r = requests.post(url, data=body, headers=headers)
                    f.write(r.text.replace(""""subject"	"predicate"	"object"\n""", ""))
                    # print("offset ", offset, " done")
                    end_t = datetime.datetime.now()
                    print("Query idx: ", (offset/batch_size), "/",batches_count," time=", end_t - start_t, " sec.")
                except  Exception as e:
                    print("Exception",e)
            q_end_t = datetime.datetime.now()
            print("total time ", q_end_t - q_start_t, " sec.")
                # print(r.text)