USE codebase;

-- JAVASCRIPT
INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'javascript','train',      assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/javascript/train/*.parquet','Parquet');

INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'javascript','validation', assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/javascript/validation/*.parquet','Parquet');

-- GO
INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'go','train',      assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/go/train/*.parquet','Parquet');

INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'go','validation', assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/go/validation/*.parquet','Parquet');

-- PYTHON
INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'python','train',      assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/python/train/*.parquet','Parquet');

INSERT INTO practice_codes (lang, split, uid, code)
SELECT 'python','validation', assumeNotNull(uid), assumeNotNull(code)
FROM file('workspace/parquet/cleaned/python/validation/*.parquet','Parquet');
