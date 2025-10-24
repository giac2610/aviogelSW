import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IonicModule, ModalController } from '@ionic/angular';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-keyboard-modal',
  templateUrl: './keyboard-modal.component.html',
  styleUrls: ['./keyboard-modal.component.scss'],
  standalone: true,
  imports: [IonicModule, CommonModule, FormsModule]
})
export class KeyboardModalComponent implements OnInit {

  // Questi valori vengono passati dal componente virtual-keyboard
  @Input() layout: 'numeric' | 'alphanumeric' = 'alphanumeric';
  @Input() initialValue: string = '';

  value: string = '';
  isShiftActive = false;
  keyboardLayout: string[][] = [];

  // Definizioni Layout
  numericLayout: string[][] = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['.', '0', 'BACKSPACE'] // Aggiunto punto per i decimali
  ];

  alphaLayout: string[][] = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['SHIFT', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'BACKSPACE'],
    ['SPACE', 'ENTER']
  ];

  constructor(private modalCtrl: ModalController) { }

  ngOnInit() {
    this.value = this.initialValue;
    this.updateKeyboardLayout();
  }

  updateKeyboardLayout(): void {
    this.keyboardLayout = (this.layout === 'numeric') ? this.numericLayout : this.alphaLayout;
  }

  // Gestisce la pressione di un tasto virtuale
  onKeyPress(key: string): void {
    switch (key) {
      case 'BACKSPACE':
        this.value = this.value.slice(0, -1);
        break;
      case 'SHIFT':
        this.isShiftActive = !this.isShiftActive;
        break;
      case 'SPACE':
        this.value += ' ';
        break;
      case 'ENTER':
        this.confirm(); // Chiude e salva
        return;
      default:
        // Aggiunge il carattere
        const char = this.isShiftActive ? key.toUpperCase() : key.toLowerCase();
        this.value += char;
        // Disattiva lo shift dopo un tasto
        if (this.isShiftActive) {
          this.isShiftActive = false;
        }
        break;
    }
  }

  // Funzione helper per visualizzare i tasti
  getDisplayKey(key: string): string {
    if (key === 'SPACE') return ''; // Gestito da CSS
    if (key === 'BACKSPACE' || key === 'SHIFT' || key === 'ENTER') return ''; // Gestito da Icone
    return this.isShiftActive ? key.toUpperCase() : key.toLowerCase();
  }

  // Chiude il modal e restituisce il valore aggiornato
  confirm(): void {
    this.modalCtrl.dismiss(this.value, 'confirm');
  }

  // Chiude il modal senza salvare
  cancel(): void {
    this.modalCtrl.dismiss(null, 'cancel');
  }
}
