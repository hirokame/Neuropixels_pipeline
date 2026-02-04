session_name = 'Y:\Ayano\TDT_photometry\Tanks\k00_k00_k1818-251119-140644\Mouse-251208-142347';
Save_univ_dir0 = 'C:\Users\kouhi\Desktop\k00_k00_k1818-251119-140644\Mouse-251208-142347';
Save_univ_dir1 = 'C:\Users\kouhi\Desktop\k00_k00_k1818-251119-140644\Mouse-251208-142347';
Save_univ_dir2 = 'C:\Users\kouhi\Desktop\k00_k00_k1818-251119-140644\Mouse-251208-142347';
Stem_Dir = 'C:\Users\kouhi\Desktop';
reload = 0;
chunky_or_not = 0;

TDT_demod(session_name,Save_univ_dir0,Save_univ_dir1,reload);
TDT_dFF_stage2(session_name,Stem_Dir,Save_univ_dir0,Save_univ_dir1,Save_univ_dir2,chunky_or_not)
