CREATE TABLE my_review_table (
    date varchar2(30),
    reviewID varchar2(30) PRIMARY KEY,
    reviewerID varchar2(30),
    reviewContent varchar2(800),
    rating INT,
    usefulCount INT,
    coolCount INT,
    funnyCount INT,
    flagged varchar2(3),
    restaurantID varchar2(30)
);

INSERT INTO my_review_table SELECT * FROM review;

CREATE TABLE my_reviewer_table (
    reviewerID varchar2(30),
    name varchar2(30),
    location varchar2(30),
    yelpJoinDate varchar2(10),
    friendCount INT,
    reviewCount INT,
    firstCount INT,
    usefulCount INT,
    coolCount INT,
    funnyCount INT,
    complimentCount INT,
    tipCount INT,
    fanCount INT
);

INSERT INTO my_reviewer_table SELECT * FROM reviewer;



ALTER TABLE my_reviewer_table ADD COLUMN reviewer_flagged varchar2(3);

UPDATE my_reviewer_table
SET reviewer_flagged = 'N'
WHERE reviewerID IN (
    SELECT ER.reviewerID
    FROM my_review_table R, my_reviewer_table ER
    WHERE R.reviewerID = ER.reviewerID
      AND R.flagged = 'N'
    );

UPDATE my_reviewer_table
SET reviewer_flagged = 'Y'
WHERE reviewerID IN (
    SELECT ER.reviewerID
    FROM my_review_table R, my_reviewer_table ER
    WHERE R.reviewerID = ER.reviewerID
      AND R.flagged = 'Y'
    );

UPDATE my_reviewer_table
SET reviewer_flagged = 'N'
WHERE reviewer_flagged IS NULL;



ALTER TABLE my_reviewer_table ADD COLUMN posReviewCount INT;
ALTER TABLE my_reviewer_table ADD COLUMN negReviewCount INT;

WITH T AS (
    SELECT COUNT(*) AS posCount, R.reviewerID
    FROM my_review_table R, my_reviewer_table ER
    WHERE R.rating > 3 AND ER.reviewerID = R.reviewerID
    GROUP BY R.reviewerID
    )
UPDATE my_reviewer_table
SET posReviewCount = (
    SELECT posCount 
    FROM T 
    WHERE my_reviewer_table.reviewerID = T.reviewerID);

UPDATE my_reviewer_table
SET posReviewCount = 0
WHERE posReviewCount IS NULL;


WITH T AS (
    SELECT COUNT(*) AS negCount, R.reviewerID
    FROM my_review_table R, my_reviewer_table ER
    WHERE R.rating < 3 AND ER.reviewerID = R.reviewerID
    GROUP BY R.reviewerID
    )
UPDATE my_reviewer_table
SET negReviewCount = (
    SELECT negCount
    FROM T
    WHERE my_reviewer_table.reviewerID = T.reviewerID);

UPDATE my_reviewer_table
SET negReviewCount = 0
WHERE negReviewCount IS NULL;



-- CREATE TABLE overall AS
--     SELECT *
--     FROM my_review_table
--         LEFT JOIN my_reviewer_table
--         USING (reviewerID);

CREATE TABLE combined AS
    SELECT *
    FROM my_review_table
        INNER JOIN my_reviewer_table
            USING (reviewerID);