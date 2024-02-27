most_occ_envs = {'S:1': 'Single_neighbor', 'L:2': 'Linear', 'A:2': 'Angular', 'TL:3': 'Trigonal_plane',

                      'TY:3': 'Trigonal_non-coplanar', 'TS:3': 'T-shaped', 'T:4': 'Tetrahedron', 'S:4': 'Square_plane',
                      'SY:4': 'Square_non-coplanar', 'SS:4': 'See-Saw', 'PP:5': 'Pentagonal_plane',

                      'S:5': 'Square_pyramid', 'T:5': 'Trigonal_bipyramid', 'O:6': 'Octahedral',
                      'T:6': 'Trigonal_prism',
                      'PP:6': 'Pentagonal_pyramid', 'PB:7': 'Pentagonal_bipyramid', 'ST:7': 'Square_faced_capped_TP',

                      'ET:7': 'End_trigonal_faced_capped_TP', 'FO:7': 'Faced_capped_octahedron', 'C:8': 'Cube',
                      'SA:8': 'Square_antiprism', 'SBT:8': 'Square-face_bicapped_TP',

                      'TBT:8': 'Triangular-face_bicapped_TP', 'DD:8': 'Dodecahedron_WTF',
                      'DDPN:8': 'Dodcahedron_WTF_p2345', 'HB:8': 'Hexagonal_bipyramid', 'BO_1:8': 'Bicapped_octahedron',

                      'BO_2:8': 'Bicapped_oct_OAC', 'BO_3:8': 'Bicapped_oct_OEC', 'TC:9': 'Triangular_cupola',
                      'TT_1:9': 'Tricapped_TP_TSF', 'TT_2:9': 'T_TP_TSF', 'TT_3:9': 'T_TP_OSF',
                      'HD:9': 'Heptagonal_dipyramid', 'TI:9': 'TI9', 'SMA:9': 'SMA9',

                      'SS:9': 'SS9', 'TO_1:9': 'TO19', 'TO_2:9': 'TO29', 'TO_3:9': 'TO3_9', 'PP:10': 'Pentagonal_prism',
                      'PA:10': 'Pentagonal_antiprism', 'SBSA:10': 'S-fBSA', 'MI:10': 'MI', 'S:10': 'S10',
                      'H:10': 'Hexadec',

                      'BS_1:10': 'BCSP_of', 'BS_2:10': 'BCSP_af', 'TBSA:10': 'TBSA',

                      'PCPA:11': 'PCPA', 'H:11': 'HDech', 'SH:11': 'SPHend', 'CO:11': 'Cs-oct', 'DI:11': 'Dimmi_icso',
                      'I:12': 'ICOSh', 'PBP:12': 'PBP12',

                      'TT:12': 'TT', 'C:12': 'Cuboctahedral', 'AC:12': 'ANTICUBOOCT', 'SC:12': 'SQU_cupola',
                      'S:12': 'Sphenemogena', 'HP:12': 'Hexagonal_prism', 'HA:12': 'Hexagonal_anti_prism',

                      'SH:13': 'SH13'}

motif_type_nmbrs = {name: i for i, name in enumerate(most_occ_envs.values(), 1)}
