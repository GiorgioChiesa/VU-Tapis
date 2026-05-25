
track_events = {
    "Insert gauze", "Remove gauze",
    "Insert hemostatic agens", "Remove hemostatic agens",
    "Hemolock clip on bladder pedicle attached to prostate", 
    "Hemolock clip on right pedicle", "Hemolock clip on left pedicle",
    "Metal clip on right pedicle", "Metal clip on left pedicle",
    "Hemostatic metal clip placement",
    "port placement", 
    "placement stitch for bladder stretch",
    "remove needle bladder stretch stitch",
    "visualisation of urethra opening", "grasping catheter tip ????",
    "identification and clipping of SV arteries - left", "identification and clipping of SV arteries - right",
    "needle removal DVC ligation", "v-lock", "cutting the needles", "removing the needles",
    "prosate bagging", "tighten endobag", "endobag removal",
    "stitch in DVC before apical dissection", "stitch in DVC after apical dissection", "stitch of posterior reconstruction",
    "stitch in bladder", "stitch in urethra", "tie suture", "final reinforcing suture",
    "drain placement", "removal of robotic instruments"
}
"""
object_events= {
        "Insert gauze", "Remove gauze", --> gauze
        "Insert hemostatic agens", "Remove hemostatic agens", --> hemostatic agens
        "Port placement", --> trocar ?? maybe is not detected
        "visualisation of urethra opening", --> urethra catheter
        "prosate bagging", "endobag removal", --> endobag
        "drain placement", --> drain tool
    }
"""
# Define instant events (excluding paired)
INSTANT_EVENTS = {
    # INSTANT_FRAME_EVENTS
    "Instrument swap: removal", "Instrument swap: insertion",
    "Insert gauze", "Remove gauze", "Insert hemostatic agens", "Remove hemostatic agens",
    "Inside abdomen", "Instrument insertion",
    "Adhesion removal", "Fat removal", "Remove needle bladder stretch stitch",
    "Needle removal DVC ligation", "V-lock", "Cutting the needles", "Removing the needles",
    "Threads removal", "Vessel loop removal", "Hemolock clip removal", "Endobag removal",
    "Drain placement", "Removal of robotic instruments", "Camera out of body", "Camera stop",
    "Prostate bagging",
    # MARGIN_1SEC_EVENTS
    "Unsuccesful clip placement", "Hemostatic metal clip placement",
    "Incision peritoneum - left", "Incision peritoneum - right",
    "Incision of the fascia - left", "Incision of the fascia - right",
    "Placement stitch for bladder stretch", "Start dissection",
    "Visualisation of urethra opening", "Grasping catheter tip",
    "Continue posterior dissection", "Hemolock clip on bladder pedicle attached to prostate",
    "Identify and dissect vas deferens - left", "Clip or coagulate vas deferens - left",
    "Identification and clipping of SV arteries - left",
    "Identify and dissect vas deferens - right", "Clip or coagulate vas deferens - right",
    "Identification and clipping of SV arteries - right",
    "Lift both seminal vesicles", "Incision of Denonvilliers fascia",
    "Lift right seminal vesicle", "Start dissection and cutting right pedicle",
    "Hemolock clip on right pedicle", "Metal clip on right pedicle",
    "Lift left seminal vesicle", "Start dissection and cutting left pedicle",
    "Hemolock clip on left pedicle", "Metal clip on left pedicle",
    "Start dissection DVC", "Stitch in DVC before apical dissection",
    "Transection of the urethra", "Tighten endobag",
    "Stitch in DVC after apical dissection", "Stitch of posterior reconstruction",
    "Stitch in bladder", "Stitch in urethra", "Tie suture",
    "Final reinforcing suture", "Endobag removal",
    # MARGIN_5SEC_EVENTS
    "Port placement", "Leak test"
}




mapping_events_name_to_id = {
            'Idle' : 0,
            'Out_of_body' : 1,
            'Back_inside_body' : 2,
            'Instrument_swap:_removal' : 3,
            'Instrument_swap:_insertion' : 4,
            'Insert_gauze' : 5,
            'Remove_gauze' : 6,
            'Insert_hemostatic_agens' : 7,
            'Remove_hemostatic_agens' : 8,
            'Unsuccesful_clip_placement' : 9,
            'Hemostatic_metal_clip_placement' : 10,
            'Test_image_start' : 11,
            'Test_image_stop' : 12,
            '3D_model' : 13,
            'Aiding_suture_stitch' : 14,
            'Aiding_suture_clip_placement' : 15,
            'Remove_needles' : 16,
            'Inside_abdomen' : 17,
            'Port_placement' : 18,
            'Instrument_insertion' : 19,
            'Adhesion_removal' : 20,
            'Incision_peritoneum_-_left' : 21,
            'Incision_peritoneum_-_right' : 22,
            'Fat_removal' : 23,
            'Incision_of_the_fascia_-_left' : 24,
            'Incision_of_the_fascia_-_right' : 25,
            'Placement_stitch_for_bladder_stretch' : 26,
            'Remove_needle_bladder_stretch_stitch' : 27,
            'Start_dissection' : 28,
            'Visualisation_of_urethra_opening' : 29,
            'Grasping_catheter_tip' : 30,
            'Continue_posterior_dissection' : 31,
            'Hemolock_clip_on_bladder_pedicle_attached_to_prostate' : 32,
            'Identify_and_dissect_vas_deferens_-_left' : 33,
            'Clip_or_coagulate_vas_deferens_-_left' : 34,
            'Identification_and_clipping_of_SV_arteries_-_left' : 35,
            'Identify_and_dissect_vas_deferens_-_right' : 36,
            'Clip_or_coagulate_vas_deferens_-_right' : 37,
            'Identification_and_clipping_of_SV_arteries_-_right' : 38,
            'Lift_both_seminal_vesicles' : 39,
            'Incision_of_Denonvilliers_fascia' : 40,
            'Lift_right_seminal_vesicle' : 41,
            'Start_dissection_and_cutting_right_pedicle' : 42,
            'Hemolock_clip_on_right_pedicle' : 43,
            'Metal_clip_on_right_pedicle' : 44,
            'Lift_left_seminal_vesicle' : 45,
            'Start_dissection_and_cutting_left_pedicle' : 46,
            'Hemolock_clip_on_left_pedicle' : 47,
            'Metal_clip_on_left_pedicle' : 48,
            'Start_dissection_DVC' : 49,
            'Stitch_in_DVC_before_apical_dissection' : 50,
            'Needle_removal_DVC_ligation' : 51,
            'Transection_of_the_urethra' : 52,
            'Prostate_bagging' : 53,
            'Tighten_endobag' : 54,
            'Stitch_in_DVC_after_apical_dissection' : 55,
            'Stitch_of_posterior_reconstruction' : 56,
            'Stitch_in_bladder' : 57,
            'Stitch_in_urethra' : 58,
            'Tie_suture' : 59,
            'V-lock' : 60,
            'Final_reinforcing_suture' : 61,
            'Leak_test' : 62,
            'Cutting_the_needles' : 63,
            'Threads_removal' : 64,
            'Vessel_loop_removal' : 65,
            'Hemolock_clip_removal' : 66,
            'Endobag_removal' : 67,
            'Drain_placement' : 68,
            'Removal_of_robotic_instruments' : 69,
            'Camera_out_of_body' : 70,
            'Camera_stop' : 71,
            'Removing_the_needles' : 16,
            'Removing_needles' : 16,
}

mapping_phases_name_to_id = {
            'Idle' : 0,
            'Port_insertion_and_surgical_access' : 1,
            'Bladder_detachment' : 2,
            'Lymphadenectomy_L' : 3,
            'Lymphadenectomy_R' : 4,
            'Endopelvic_fascia_incision' : 5,
            'Bladder_neck_dissection' : 6,
            'Vas_deferens_and_seminal_vesicles' : 7,
            'Dissection_posterior_space_between_prostate_and_rectum' : 8,
            'Right_lateral_dissection_of_the_prostate' : 9,
            'Left_lateral_dissection_of_the_prostate' : 10,
            'DVC_dissection' : 11,
            'Apical_dissection' : 12,
            'Posterior_reconstruction' : 13,
            'VU-anastomosis' : 14,
            'Specimen_and_instrument_removal' : 15,
            'End_of_operation' : 16,
}

mapping_events_id_to_name = {v: k for k, v in mapping_events_name_to_id.items()}
mapping_phases_id_to_name = {v: k for k, v in mapping_phases_name_to_id.items()}


class_mapping = {
    
    # INCISION - Incisioni e aperture di piani tissutali
    'Incision_peritoneum_-_left': 'INCISION',
    'Incision_peritoneum_-_right': 'INCISION',
    'Incision_of_the_fascia_-_left': 'INCISION',
    'Incision_of_the_fascia_-_right': 'INCISION',
    'Incision_of_Denonvilliers_fascia': 'INCISION',
    
    # DISSECTION - Dissezione e separazione di strutture anatomiche
    'Start_dissection': 'DISSECTION',
    'Continue_posterior_dissection': 'DISSECTION',
    'Start_dissection_DVC': 'DISSECTION',
    'Start_dissection_and_cutting_right_pedicle': 'DISSECTION',
    'Start_dissection_and_cutting_left_pedicle': 'DISSECTION',
    'Identify_and_dissect_vas_deferens_-_left': 'DISSECTION',
    'Identify_and_dissect_vas_deferens_-_right': 'DISSECTION',
    'Transection_of_the_urethra': 'DISSECTION',
    'Adhesion_removal': 'DISSECTION',
    'Fat_removal': 'DISSECTION',
    
    # HEMOSTASIS - Arresto dell'emorragia tramite clips e coagulazione
    'Hemostatic_metal_clip_placement': 'HEMOSTASIS',
    'Unsuccesful_clip_placement': 'HEMOSTASIS',
    'Hemolock_clip_on_bladder_pedicle_attached_to_prostate': 'HEMOSTASIS',
    'Hemolock_clip_on_right_pedicle': 'HEMOSTASIS',
    'Hemolock_clip_on_left_pedicle': 'HEMOSTASIS',
    'Metal_clip_on_right_pedicle': 'HEMOSTASIS',
    'Metal_clip_on_left_pedicle': 'HEMOSTASIS',
    'Clip_or_coagulate_vas_deferens_-_left': 'HEMOSTASIS',
    'Clip_or_coagulate_vas_deferens_-_right': 'HEMOSTASIS',
    'Hemolock_clip_removal': 'HEMOSTASIS',
    
    # MATERIAL HANDLING - Inserimento e rimozione di materiali (garze, drenaggi, endobag)
    'Insert_gauze': 'MATERIAL_HANDLING',
    'Remove_gauze': 'MATERIAL_HANDLING',
    'Insert_hemostatic_agens': 'MATERIAL_HANDLING',
    'Remove_hemostatic_agens': 'MATERIAL_HANDLING',
    'Prostate_bagging': 'MATERIAL_HANDLING',
    'Tighten_endobag': 'MATERIAL_HANDLING',
    'Endobag_removal': 'MATERIAL_HANDLING',
    'Vessel_loop_removal': 'MATERIAL_HANDLING',
    'Threads_removal': 'MATERIAL_HANDLING',
    'Drain_placement': 'MATERIAL_HANDLING',
    # INSTRUMENT HANDLING - Gestione e scambio di strumenti chirurgici
    'Instrument_swap:_removal': 'MATERIAL_HANDLING',
    'Instrument_swap:_insertion': 'MATERIAL_HANDLING',
    'Instrument_insertion': 'MATERIAL_HANDLING',
    'Removal_of_robotic_instruments': 'MATERIAL_HANDLING',
    'Port_placement': 'MATERIAL_HANDLING',
    
    # SUTURING - Azioni di sutura e ricostruzione
    'Aiding_suture_stitch': 'SUTURING',
    'Aiding_suture_clip_placement': 'SUTURING',
    'Placement_stitch_for_bladder_stretch': 'SUTURING',
    'Remove_needle_bladder_stretch_stitch': 'SUTURING',
    'Stitch_in_DVC_before_apical_dissection': 'SUTURING',
    'Stitch_in_DVC_after_apical_dissection': 'SUTURING',
    'Stitch_of_posterior_reconstruction': 'SUTURING',
    'Stitch_in_bladder': 'SUTURING',
    'Stitch_in_urethra': 'SUTURING',
    'Tie_suture': 'SUTURING',
    'V-lock': 'SUTURING',
    'Final_reinforcing_suture': 'SUTURING',
    'Needle_removal_DVC_ligation': 'SUTURING',
    'Cutting_the_needles': 'SUTURING',
    'Removing_the_needles': 'SUTURING',
    'Remove_needles': 'SUTURING',
    'Removing_needles': 'SUTURING',
    
    # ANATOMICAL EXPLORATION - Identificazione, visualizzazione e manovra di strutture
    'Visualisation_of_urethra_opening': 'ANATOMICAL_EXPLORATION',
    'Grasping_catheter_tip': 'ANATOMICAL_EXPLORATION',
    'Lift_both_seminal_vesicles': 'ANATOMICAL_EXPLORATION',
    'Lift_right_seminal_vesicle': 'ANATOMICAL_EXPLORATION',
    'Lift_left_seminal_vesicle': 'ANATOMICAL_EXPLORATION',
    'Identification_and_clipping_of_SV_arteries_-_left': 'ANATOMICAL_EXPLORATION',
    'Identification_and_clipping_of_SV_arteries_-_right': 'ANATOMICAL_EXPLORATION',
    
    
    # VERIFICATION - Test di verifica e controlli diagnostici
    'Leak_test': 'VERIFICATION',
    '3D_model': 'VERIFICATION',
    
    # SYSTEM CONTROL - Controllo della telecamera e setup di sistema
    'Camera_out_of_body': 'SYSTEM_CONTROL',
    'Camera_stop': 'SYSTEM_CONTROL',
    'Test_image_start': 'SYSTEM_CONTROL',
    'Test_image_stop': 'SYSTEM_CONTROL',
    'Inside_abdomen': 'SYSTEM_CONTROL',
    'Out_of_body': 'SYSTEM_CONTROL',
    'Back_inside_body': 'SYSTEM_CONTROL',
    
    # MISC - Idle e azioni varie
    'Idle': 'MISC',

}

classes = {
    "INCISION":0,
    "DISSECTION":1,
    "HEMOSTASIS":2,
    "MATERIAL_HANDLING":3,
    "SUTURING":4,
    "ANATOMICAL_EXPLORATION":5,
    "VERIFICATION":6,
    "SYSTEM_CONTROL":7,
    "MISC":8
}

num_classes = len(classes)



if __name__ == "__main__":
    from glob import glob
    import os
    import pandas as pd
    import json
    from tqdm import tqdm
    from detectron2.utils.file_io import PathManager as pathmgr

    # csv_files = glob(os.path.join("/data/orsi_tensors/*/*/", "*.csv"))
    mapping_id2idclass = {mapping_events_name_to_id[k]: int(classes[v]) for k, v in class_mapping.items()}
    # mapping_id2nameclass = {mapping_events_name_to_id[k]: v for k, v in class_mapping.items()}
    # for csv_file in csv_files:
    #     df = pd.read_csv(csv_file)
    #     print(f"File: {csv_file}")
    #     try:
    #         # Add classes_name column using class_mapping dict
    #         df['classes_id'] = df['event_id'].map(mapping_id2idclass)
    #         df['classes_name'] = df['event_id'].map(mapping_id2nameclass)
            
            
    #         # print(df.head())
    #         # Save the updated DataFrame back to CSV
    #         df.to_csv(csv_file, index=False)
    #     except Exception as e:
    #         print(f"Error processing {csv_file}: {e}")
    
    
    json_files = glob(os.path.join("/data/coco", "*.json"))
    for json_file in tqdm(json_files, desc="Processing JSON files", total=len(json_files)):
        print(f"File: {json_file}")
        with pathmgr.open(json_file, "r") as f:
            data = json.load(f)
        data["classes_categories"] = [{"id": id_class, "name": clas, "description": ""} for clas, id_class in classes.items() ]

        ann = data["annotations"]
        for a in tqdm(ann, desc="annot:", total=len(ann), leave=False):
            a["classes"] = mapping_id2idclass.get(a["steps"], -1)
        
        with pathmgr.open(json_file, "w") as f:
            json.dump(data, f)

    
