import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { LedService } from './led.service';
@Injectable({
  providedIn: 'root'
})
export class MotorsControlService {
  private apiUrl = `http://${window.location.hostname}:8000/motors/`
  

  constructor(
    private http: HttpClient, 
    private ledService: LedService
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
  simulate(): Observable<any> {
    const url = `${this.apiUrl}simulate/`;
    return this.http.post<any>(url, null);
  }

  goHome(body: any): Observable<any> {
    const url = `${this.apiUrl}home/`;
    // this.ledService.startWaveEffect().subscribe({
    //   next: () => console.log('wave effect avviato'),
    //   error: () => console.error('Errore nell\'avviare il wavew effect')
    // });
    return this.http.post<any>(url, body);
  }

  syringeExtrusionStart(): Observable<any> {
    const url = `${this.apiUrl}syringe/extrusion/start/`;
    return this.http.post<any>(url, null);
  }
}
