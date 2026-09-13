import cv2
import easyocr
from ultralytics import YOLO
import datetime
import sqlite3

conn = sqlite3.connect("DataBase.db")
cursor = conn.cursor()
cursor.execute("create table if not exists info (Id INTEGER PRIMARY KEY, Vehicle_Type TEXT, Seatbelt TEXT, Mobile TEXT, Plate_Number TEXT, Date TEXT)")
conn.commit()

vehicle = {}

reader = easyocr.Reader(['en'])

vehicle_model = YOLO('vehicle.pt',task='detect')
seatbelt_model = YOLO('seatbelt.pt',task='detect')
mobile_model = YOLO('mobile.pt',task='detect')
plate_model = YOLO('plate.pt',task='detect')

cam = cv2.VideoCapture(0)

while True:
    ret , frame = cam.read()
    if not ret:
        break
    results = vehicle_model.track(frame,persist=True,tracker='botsort.yaml',verbose=False)
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            if conf < 0.4:
                continue

            if box.id is None:
                continue

            tracked_id = int(box.id)

            if tracked_id not in vehicle:

                vehicle[tracked_id] = {"Type":"unknown",
                                       "Seatbelt":"unknown",
                                       "Plate Number":None,
                                       "Mobile":"unknown",
                                       "Date":datetime.datetime.now().isoformat(),
                                       "DataBase":"No"}

            if class_id == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                vehicle[tracked_id]['Type'] = "Car"

            elif class_id == 7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                vehicle[tracked_id]['Type'] = "Truck"

            vehicle_crop = frame[y1:y2,x1:x2]

            seatbelt_result = seatbelt_model.predict(vehicle_crop,verbose=False)
            plate_result = plate_model.predict(vehicle_crop, verbose=False)
            mobile_result = mobile_model.predict(vehicle_crop,verbose=False)

            for sb in seatbelt_result[0].boxes:
                sx1,sy1,sx2,sy2 = map(int,sb.xyxy[0])
                seatbelt_class = int(sb.cls[0])

                sx1 += x1
                sy1 += y1
                sx2 += x1
                sy2 += y1

                if seatbelt_class == 0 :
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                    vehicle[tracked_id]['Seatbelt'] = "No"

                elif seatbelt_class == 1 :
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                    vehicle[tracked_id]['Seatbelt'] = "Yes"

            for mb in mobile_result[0].boxes:
                mx1,my1,mx2,my2 = map(int,mb.xyxy[0])
                mobile_class = int(mb.cls[0])

                mx1 += x1
                my1 += y1
                mx2 += x1
                my2 += y1

                if mobile_class == 0 :
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
                    vehicle[tracked_id]['Mobile'] = "Yes"

            for pl in plate_result[0].boxes:
                px1,py1,px2,py2 = map(int,pl.xyxy[0])

                plate_crop = vehicle_crop[py1:py2,px1:px2]

                px1 += x1
                py1 += y1
                px2 += x1
                py2 += y1

                if vehicle[tracked_id]['Plate Number'] is None :
                    ocr = reader.readtext(plate_crop)

                    if ocr:
                        text = ocr[0][1]

                        vehicle[tracked_id]['Plate Number'] = text

                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

            if vehicle[tracked_id]['DataBase'] == "No" and vehicle[tracked_id]['Plate Number'] is not None:
                cursor.execute("insert into info(Vehicle_Type,Seatbelt,Mobile,Plate_Number,Date) values(?,?,?,?,?)",(vehicle[tracked_id]['Type'],vehicle[tracked_id]['Seatbelt'],vehicle[tracked_id]['Mobile'],vehicle[tracked_id]['Plate Number'],vehicle[tracked_id]['Date']))
                conn.commit()
                vehicle[tracked_id]['DataBase'] = "Yes"

    cv2.imshow('webcam',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
conn.close()