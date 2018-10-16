create database mahout;

use mahout;

CREATE TABLE taste_preferences (
user_id BIGINT NOT NULL,
item_id BIGINT NOT NULL,
preference FLOAT NOT NULL,
PRIMARY KEY (user_id, item_id),
INDEX (user_id),
INDEX (item_id)
);

load data local infile "/data/ratings.dat" into
table taste_preferences fields terminated by '::'(user_id, item_id, preference);
