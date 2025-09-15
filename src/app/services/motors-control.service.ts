import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class MotorsControlService {
  private apiUrl = `http://${window.location.hostname}:8000/motors/`
  

  constructor(
    private http: HttpClient, 
  ) { }

  updateSettings(body: any): Observable<any> {
    const url = `${this.apiUrl}update/`
    return this.http.post<any>(url, body);
  }

  moveMotor(body: any): Observable<any> {
    const url = `${this.apiUrl}move/`
    return this.http.post<any>(url, body);
  }
  // ferma tutti i motori
  stopMotor(): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}stop/`, null);
  }
  saveSettings(body: any): Observable<any> {
    const url = `${this.apiUrl}save/`
    return this.http.post<any>(url, body);
  }

  goHome(body: any): Observable<any> {
    const url = `${this.apiUrl}home/`;
    return this.http.post<any>(url, body);
  }

  syringeExtrusionStart(): Observable<any> {
    const url = `${this.apiUrl}syringe/extrusion/start/`;
    return this.http.post<any>(url, null);
  }
}
