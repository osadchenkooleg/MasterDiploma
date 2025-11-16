USE codebase;

-- Позитиви: збіги за нормалізованим MD5 в межах мови
INSERT INTO practice_pairs (split, a_lang, a_uid, b_lang, b_uid, label, notes)
SELECT 'valid', c1.lang, c1.uid, c2.lang, c2.uid, 1, 'norm_md5_dup'
FROM practice_codes c1
JOIN practice_codes c2
  ON c1.lang = c2.lang
 AND c1.uid  != c2.uid
 AND c1.code_norm_md5 = c2.code_norm_md5
WHERE c1.lang IN ('javascript','go','python')
LIMIT 2000;

-- Негативи: випадкові пари (за потреби додай умови різних задач, якщо є метадані)
INSERT INTO practice_pairs (split, a_lang, a_uid, b_lang, b_uid, label, notes)
SELECT 'valid', c1.lang, c1.uid, c2.lang, c2.uid, 0, 'random_neg'
FROM practice_codes c1
JOIN practice_codes c2
  ON c1.uid != c2.uid
WHERE c1.lang IN ('javascript','go','python')
  AND c2.lang IN ('javascript','go','python')
LIMIT 2000;

-- Перевірка
SELECT label, count() AS n FROM practice_pairs GROUP BY label ORDER BY label;
