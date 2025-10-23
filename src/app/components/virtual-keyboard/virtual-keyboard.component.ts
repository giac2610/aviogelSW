import { CommonModule } from '@angular/common';
import { Component, Input, AfterViewInit, ViewChild, forwardRef } from '@angular/core';
import { IonicModule, IonInput } from '@ionic/angular';
import * as KioskBoard from 'kioskboard';
// Import necessary modules for ngModel
import { ControlValueAccessor, NG_VALUE_ACCESSOR, FormsModule } from '@angular/forms';

@Component({
  selector: 'app-virtual-keyboard',
  // Add FormsModule to imports for standalone components
  imports: [IonicModule, CommonModule, FormsModule],
  templateUrl: './virtual-keyboard.component.html',
  styleUrls: ['./virtual-keyboard.component.scss'],
  // Provider to connect the component with Angular's Forms API
  providers: [
    {
      provide: NG_VALUE_ACCESSOR,
      useExisting: forwardRef(() => VirtualKeyboardComponent),
      multi: true
    }
  ]
})
// Implement the ControlValueAccessor interface
export class VirtualKeyboardComponent implements AfterViewInit, ControlValueAccessor {

  @Input() layout: 'alpha' | 'numpad' = 'alpha';
  @Input() placeholder = '';
  @Input() type = 'text';
  @Input() label = ''; // Add label as an Input

  @ViewChild('kbInput', { static: true }) input!: IonInput;

  // Property to hold the component's value
  value: any;
  isDisabled = false;

  // --- Implementation of ControlValueAccessor ---

  // Placeholder for the function that should be called when the value changes
  onChange: any = () => {};
  // Placeholder for the function that should be called when the component is touched
  onTouched: any = () => {};

  /**
   * Writes a new value from the form model into the component.
   */
  writeValue(value: any): void {
    this.value = value;
  }

  /**
   * Registers a callback function to be called when the component's value changes.
   */
  registerOnChange(fn: any): void {
    this.onChange = fn;
  }

  /**
   * Registers a callback function to be called when the component is "touched".
   */
  registerOnTouched(fn: any): void {
    this.onTouched = fn;
  }

  /**
   * This function is called when the control's disabled state changes.
   */
  setDisabledState?(isDisabled: boolean): void {
    this.isDisabled = isDisabled;
  }

  // --- Component Logic ---

  ngAfterViewInit() {
    this.input.getInputElement().then(inputEl => {
      // Listener focus: apri la tastiera
      inputEl.addEventListener('focus', () => {
        this.openKeyboard(inputEl);
      });
    });
  }
  
  // This method is called by the template when the input value changes
  onValueChange(event: any) {
    const newValue = event.target.value;
    this.value = newValue;
    // Notify Angular that the value has changed
    this.onChange(newValue);
  }

  private openKeyboard(inputEl: HTMLInputElement) {
    const numericKeys = [
  { "0": "1", "1": "2", "2": "3" },
  { "0": "4", "1": "5", "2": "6" },
  { "0": "7", "1": "8", "2": "9" },
  { "0": ",", "1": "0", "2": "{bksp}" }
];
const alphaKeys = [
  { "0": "1", "1": "2", "2": "3" },
  { "0": "4", "1": "5", "2": "6" },
  { "0": "7", "1": "8", "2": "9" },
  { "0": ",", "1": "0", "2": "{bksp}" }
];
const config: any = {
  language: 'it',
  theme: 'dark',
  allowRealKeyboard: false,
  allowMobileKeyboard: false,
  cssAnimations: true,
  cssAnimationsDuration: 360,
  cssAnimationsStyle: 'slide-up',
  keysAllowNumeric: this.layout === 'numpad' ? false : true,
  keysArrayOfObjects: this.layout === 'alpha' ? alphaKeys : numericKeys
};
    KioskBoard.run(inputEl, config);
  }
}