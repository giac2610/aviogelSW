import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MotorsControlService {
  private apiUrl = `http://${window.location.hostname}:8000/motors/` 
  
  constructor(private http: HttpClient) { }
  
  // Ottieni le impostazioni dei motori
  updateSettings(body: any):Observable<any>{
    const url = `${this.apiUrl}update/`
    return this.http.post<any>(url, body);
    }
  
  // richiesta per muovere il motore
  // il body deve essere un oggetto con le seguenti proprietà:
  // motor: string, // il nome del motore (syringe, extruder, conveyor)
  // distance: number, // la distanza da percorrere
  moveMotor(body: any):Observable<any> {
    const url = `${this.apiUrl}move/`
    return this.http.post<any>(url, body);
  }
  // ferma tutti i motori
  stopMotor():Observable<any>{
    return this.http.post<any>(`${this.apiUrl}stop/`, null);
  }
  stopMotors(): Observable<any> {
    const url = `${this.apiUrl}stop/`;
    return this.http.post<any>(url, null);
  }
  saveSettings(body: any):Observable<any>{
    const url = `${this.apiUrl}save/`
    return this.http.post<any>(url, body);
  }
  simulate(): Observable<any> {
    const url = `${this.apiUrl}simulate/`;
    return this.http.post<any>(url, null);
  }
}
