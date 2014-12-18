drop database if exists joyce;
CREATE DATABASE joyce
  DEFAULT CHARACTER SET utf8
  DEFAULT COLLATE utf8_general_ci;

connect joyce;

drop table if exists documents;
CREATE TABLE documents
(
document_id int not null auto_increment,
author varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci not null,
date datetime,
title varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci not null, 
type varchar(250) CHARACTER SET utf8 COLLATE utf8_unicode_ci not null, 
document longtext CHARACTER SET utf8 COLLATE utf8_unicode_ci not null,
primary key (document_id)
); 

drop table if exists segments;
CREATE TABLE segments
(
segment_id int not null auto_increment,
document_id int not null,
segment_text text CHARACTER SET utf8 COLLATE utf8_unicode_ci not null,
primary key (segment_id)
);

drop table if exists analysis;
CREATE TABLE analysis
(
run_id int not null auto_increment,
parameters text not null, 
primary key (run_id)
);

