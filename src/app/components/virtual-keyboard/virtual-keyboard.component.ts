import { CommonModule } from '@angular/common';
import { Component, Input, forwardRef, ChangeDetectorRef } from '@angular/core';
import { IonicModule, ModalController } from '@ionic/angular';
import { ControlValueAccessor, NG_VALUE_ACCESSOR} from '@angular/forms';
import { KeyboardModalComponent } from '../../keyboard-modal/keyboard-modal.component';

@Component({
  selector: 'app-virtual-keyboard',
  imports: [
    IonicModule,
    CommonModule,
  ],
  templateUrl: './virtual-keyboard.component.html',
  styleUrls: ['./virtual-keyboard.component.scss'],
  standalone: true,
  providers: [
    {
      provide: NG_VALUE_ACCESSOR,
      useExisting: forwardRef(() => VirtualKeyboardComponent),
      multi: true
    }
  ]
})
export class VirtualKeyboardComponent implements ControlValueAccessor {

  @Input() layout: 'numeric' | 'alphanumeric' = 'alphanumeric';
  @Input() placeholder = '';
  @Input() type: 'text' | 'number' | 'tel' = 'text';
  @Input() label = ''; // Riattivato l'Input per la label

  value: string = '';
  isDisabled = false;
  onChange: (value: string | number | null)=> void = () => {};
  onTouched: () => void = () => {};

  constructor(
    private modalCtrl: ModalController,
    private cdr: ChangeDetectorRef
  ) {}

  // --- Metodi ControlValueAccessor ---
  writeValue(value: any): void {
    this.value = (value !== null && value !== undefined) ? String(value) : '';
    this.cdr.markForCheck();
  }

  registerOnChange(fn: any): void {
    this.onChange = fn;
  }

  registerOnTouched(fn: any): void {
    this.onTouched = fn;
  }

  setDisabledState?(isDisabled: boolean): void {
    this.isDisabled = isDisabled;
    this.cdr.markForCheck();
  }

  // --- Gestione Apertura Modal ---
  async onInputClick(): Promise<void> {
    if (this.isDisabled) return;

    this.onTouched();

    const modal = await this.modalCtrl.create({
      component: KeyboardModalComponent,
      componentProps: {
        layout: this.layout,
        initialValue: this.value,
        inputLabel: this.label
      },
      cssClass:'keyboard-modal-bottom'
    });

    await modal.present();

    const { data, role } = await modal.onWillDismiss();

    if (role === 'confirm') { 
            let valueToEmit: string | number | null;

            if (data === null || data === undefined || data === '') {
                // Se l'utente ha cancellato o chiuso senza valore
                valueToEmit = null; 
                this.value = ''; // Aggiorna il display interno
            } 
            // Controlla se l'input DOVEVA essere un numero
            else if (this.layout === 'numeric' || this.type === 'number') { 
                
                // Tenta di convertire la stringa 'data' (es. "2") in un numero
                const num = parseFloat(data); 

                if (!isNaN(num)) {
                    // La conversione è riuscita
                    valueToEmit = num; // Emetti il NUMERO (es. 2)
                    this.value = data; // L'input interno (this.value) può rimanere stringa ("2")
                } else {
                    // L'utente ha inserito qualcosa che non è un numero (es. ".")
                    valueToEmit = null; 
                    this.value = data; 
                }
            } else {
                // Se non è numerico, emetti la stringa (es. "Testo")
                valueToEmit = data; 
                this.value = data;
            }
            
            // Chiama onChange con il tipo corretto (NUMERO o STRINGA o NULL)
            this.onChange(valueToEmit); 
            this.cdr.markForCheck();
          }
  }
}

