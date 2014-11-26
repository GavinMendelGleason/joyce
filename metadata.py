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

def connectDB(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB, **kwargs):
    db = MySQLdb.connect(host,user,passwd,db, **kwargs)
    return db

FRESHDB = """
drop table if exists documents;
CREATE TABLE documents
(
document_id int not null auto_increment,
author varchar(250) not null,
date datetime,
title varchar(250) not null, 
document LONGBLOB not null,
primary key (document_id)
); 

drop table if exists segments;
CREATE TABLE segments
(
segment_id int not null auto_increment,
document_id int not null,
segment_text text not null,
primary key (segment_id)
);

drop table if exists analysis;
CREATE TABLE analysis
(
run_id int not null auto_increment,
parameters text not null, 
primary key (run_id)
);

"""
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
    
def guess_title(path): 
    with open(path, 'rb') as f:
        for line in f: 
            sl = line.strip() 
            if not sl == '':                 
                return sl 
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
    m = re.search('\s(\w+)\s*$', chunk[0:maxsize], re.MULTILINE)
    if m: 
        return m.start()

def chunkify(doctext,maxsize=1000, underflow_limit=50):
    doctext = collapse_separators(doctext)
    doctext = remove_space(doctext)
    chunks = []
    overflow = ''
    for i in xrange(0,len(doctext),maxsize): 
        current = overflow + doctext[i:i+maxsize]
        cut_at = last_word_index(current,maxsize)
        if cut_at and cut_at > maxsize - underflow_limit:
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

    SQL = """select segment_text from segments as s, documents as d
             where s.document_id = d.document_id
             and d.type = 'train' 
             order by segment_id asc;"""
    db = connectDB() 
    cursor = db.cursor() 
    cursor.execute(SQL) 
    segments = [seg for (seg,) in cursor]
    return segments

def segment(db,docid,maxsize=1000): 
    SQL = """select document from documents 
             where document_id=%(document_id)s"""
    cursor = db.cursor()
    cursor.execute(SQL,{'document_id' : docid})
    (doctext,) = cursor.fetchone()
    res = chunkify(doctext,maxsize)
    INSERT = """insert into segments (document_id,segment_text) VALUES (%(document_id)s,%(segment_text)s)"""
    for chunk in res:
        cursor.execute(INSERT, {'document_id' : docid, 
                                'segment_text' : chunk})
    db.commit()
    return res


def get_joyce_or_not_features(): 
    SQL = """select author from documents as d, segments as s
             where d.document_id = s.document_id
             and d.type = 'train'
             order by segment_id asc;"""
    db = connectDB()
    cursor = db.cursor()
    cursor.execute(SQL)
    results = [ 1 if a == 'James Joyce' else 0 for (a,) in cursor ]
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
    args = vars(parser.parse_args())

    setupLogging() 
    
    maybeCreateDB()

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
            p = d['file']
            basedir = os.path.dirname(args['xls'])
            s = read_file(basedir + '/' + p)
            enc = d['encoding']
            d['document'] = s.decode(enc).encode('utf-8')
            cursor.execute(DOC_STATEMENT, d)

    elif args['file_pattern']:
        files = glob.glob(args['file_pattern'])

        for p in files: 
            s = read_file(p)
            # input fi
            s = s.decode(args['encoding'])
            s = s.encode('utf-8')
            d = {'year' : guess_year(p), 
                 'title' : guess_title(p), 
                 'document' : s,
                 'type' : args['type'],
                 'author' : args['author']}
            cursor.execute(DOC_STATEMENT, d)
            
        db.commit()

    # Segment previously unsegmented documents: 
    SEGMENTS = """
    select document_id from documents
    where document_id not in (select distinct document_id 
                              from segments)
    """
    documents = []
    cursor.execute(SEGMENTS)
    for (document_id,) in cursor: 
        segment(db,document_id)

    db.commit()
        
