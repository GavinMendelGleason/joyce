#!/usr/bin/env python

import xlrd 
import argparse 
import logging 
import MySQLdb
import glob
import os
from os import getenv
import re
from config import * 

def connectDB(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB, use_unicode=True, charset="utf8", **kwargs):
    db = MySQLdb.connect(host,user,passwd,db, **kwargs)
    return db

__LIB_PATH__ = getenv("HOME") + '/lib/Joycechekovgavin/'
__DEFAULT_PATH__ = __LIB_PATH__ + 'corpus/'
__LOG_PATH__ = __LIB_PATH__ + 'metadata.log'
__LOG_FORMAT__ = "%(asctime)-15s %(message)s"

def maybeCreateDB(): 
    """Create the metadata tables if they don't already exist"""
    db = connectDB()
    cursor = db.cursor() 
    cursor.connection.autocommit(True)
    tables = len(re.findall("CREATE TABLE", FRESHDB))
    length = cursor.execute("show tables")

    if not length == tables:
        cursor.execute(FRESHDB)
    
def guess_title(path,enc): 
    with open(path, 'rb') as f:
        for line in f: 
            sl = line.strip() 
            if not sl == '':                 
                return sl.decode('enc') 
        return "Unknown"

def guess_year(path): 
    date_pat = '/([0-9][0-9][0-9][0-9])[^/]*$'
    m = re.search(date_pat, path) 
    if m: 
        year = m.group(1)
        return year
    else:
        return None

def read_file(p): 
    with open (p, "rb") as myfile:
        data=myfile.read()
    return data

def setupLogging(): 
    # set up logging
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(filename=args['log'],level=logging.INFO,
                        format=__LOG_FORMAT__)

def last_word_index(chunk,maxsize):
    m = re.search('(\.|!|\?)(\w|\s)*$', chunk[0:maxsize], re.MULTILINE)
    if m: 
        return m.start() + 1
    else: 
        m = re.search('\s*\w+(\s)*$', chunk[0:maxsize], re.MULTILINE)
        if m: 
            return m.start() + 1
        else: 
            return maxsize

## No longer using underflow limit
def chunkify(doctext,maxsize=1000): #, underflow_limit=50):
    doctext = collapse_separators(doctext)
    doctext = remove_space(doctext)
    chunks = []
    overflow = ''
    for i in xrange(0,len(doctext),maxsize): 
        current = overflow + doctext[i:i+maxsize]
        cut_at = last_word_index(current,maxsize)
        if cut_at: # and cut_at > maxsize - underflow_limit:
            chunks.append(current[0:cut_at])
            overflow = current[cut_at:]
        else:
            # throw out overflow and final chunk?
            # it's probably safe...
            pass
    return chunks

def remove_space(doctext): 
    return re.sub('\s+', ' ', doctext)
    
def collapse_separators(doctext): 
    doctext = re.sub('\.\.\.\.+',' ',doctext) 
    doctext = re.sub('---+',' ',doctext)
    return doctext

def get_training_segments(): 
    """Get segments, in order."""

    SQL = """select title,segment_text 
             from segments as s, documents as d
             where s.document_id = d.document_id
             and d.type = 'train' 
             order by segment_id asc;"""
    db = connectDB() 
    cursor = db.cursor() 
    cursor.execute(SQL) 
    segments = [seg for (seg,) in cursor]
    return segments
    
def get_test_segments(): 
    """Get test segments, in order."""

    SQL = """select segment_text from segments as s, documents as d
             where s.document_id = d.document_id
             and d.type = 'test' 
             order by segment_id asc;"""
    db = connectDB() 
    cursor = db.cursor() 
    cursor.execute(SQL) 
    segments = [seg for (seg,) in cursor]
    return segments

def get_all_segments(): 
    """Get segments, in order."""

    SQL = """select segment_text from segments as s, documents as d
             where s.document_id = d.document_id
             order by segment_id asc;"""
    db = connectDB() 
    cursor = db.cursor() 
    cursor.execute(SQL) 
    segments = [seg for (seg,) in cursor]
    return segments

def segment(db,docid,maxsize=500): 
    SQL = """select title,document from documents 
             where document_id=%(document_id)s"""
    cursor = db.cursor()
    cursor.execute(SQL,{'document_id' : docid})
    (title,doctext,) = cursor.fetchone()
    try: 
        res = chunkify(doctext.decode('utf-8'),maxsize)
    except Exception: 
        print "Failed to load: "+title
    INSERT = """insert into segments (document_id,segment_text) VALUES (%(document_id)s,%(segment_text)s)"""
    for chunk in res:
        cursor.execute(INSERT, {'document_id' : docid, 
                                'segment_text' : chunk})
    db.commit()
    return res

def get_training_author_or_not_features(author='James Joyce'): 
    SQL = """select author from documents as d, segments as s
             where d.document_id = s.document_id
             and d.type = 'train'
             order by segment_id asc;"""
    db = connectDB()
    cursor = db.cursor()
    cursor.execute(SQL)
    results = [ 1 if a == author else 0 for (a,) in cursor ]
    return results

def get_all_author_or_not_features(author='James Joyce'): 
    SQL = """select author from documents as d, segments as s
             where d.document_id = s.document_id
             order by segment_id asc;"""
    db = connectDB()
    cursor = db.cursor()
    cursor.execute(SQL)
    results = [ 1 if a == author else 0 for (a,) in cursor ]
    return results

if __name__ == '__main__': 
    """We will need some code for populating tables from command line 
       here, possibly by sending in some basic parameters and a glob.
    """    
    parser = argparse.ArgumentParser(description='Allow command line uploading of files into the database.')
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--xls', help='Ignore all other switches and use an excell spreadsheet to determine parameters', default=False)
    parser.add_argument('--file-pattern', help='Regex pattern for loading files for the given author', default=None)
    parser.add_argument('--author', help='Specify the author for this traunch', default='Unknown')
    parser.add_argument('--type', help="Type of input set, 'test', 'train'", default='train')
    parser.add_argument('--encoding', help="Encoding type for import", default='latin_1')
    parser.add_argument('--seg-size', help="Size of segments", default="2000")
    args = vars(parser.parse_args())

    setupLogging() 
    
    maybeCreateDB()

    DELETE_STATEMENT="""
       DELETE from documents where title=%(title)s;
    """
    DOC_STATEMENT="""
       INSERT into documents
       (author,date,title,document,type)
       VALUES
       (%(author)s,MAKEDATE(%(year)s,1),%(title)s,%(document)s,%(type)s)
    """
    db = connectDB()
    cursor = db.cursor()
    
    if args['xls']: 
        # Need to read in from xls file.    
        book = xlrd.open_workbook(args['xls'], encoding_override="cp1252")
        # Fragile?  We could just take the first one in case renamed.
        sheet1 = book.sheet_by_name('Sheet1')
        length = sheet1.nrows
        header = sheet1.row(0)
        header_names = []
        for cell in header: 
            header_names.append(cell.value)

        for i in range(1, length): 
            d = {}
            for j in range(0,len(header_names)): 
                d[header_names[j].lower()] = sheet1.row(i)[j].value
            print "processing file %s "% d['file']
            p = d['file']
            basedir = os.path.dirname(args['xls'])
            s = read_file(basedir + '/' + p)
            print "with encoding %s "% d['encoding']
            enc = d['encoding']
            d['document'] = s.decode(enc)
            print type(d['title'])
            print type(d['document'])
            cursor.execute(DELETE_STATEMENT, {'title' : d['title']})
            cursor.execute(DOC_STATEMENT, d)

    elif args['file_pattern']:
        files = glob.glob(args['file_pattern'])

        for p in files: 
            s = read_file(p)
            # input fi
            s = s.decode(args['encoding'])
            s = s.encode('utf-8')
            d = {'year' : guess_year(p), 
                 'title' : guess_title(p,args['encoding']), 
                 'document' : s,
                 'type' : args['type'],
                 'author' : args['author']}
            cursor.execute(DOC_STATEMENT, d)
            
        db.commit()

    ELIM_SEGMENTS = """ 
    delete from segments;
    """
    cursor.execute(ELIM_SEGMENTS)

    # Segment previously unsegmented documents: 
    SEGMENTS = """
    select document_id from documents
    where document_id not in (select distinct document_id 
                              from segments)
    """
    documents = []
    cursor.execute(SEGMENTS)
    for (document_id,) in cursor: 
        segment(db,document_id,maxsize=int(args['seg_size']))

    db.commit()
        
