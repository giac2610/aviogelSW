import { Component, Input, AfterViewInit, ElementRef, OnDestroy, Renderer2 } from '@angular/core';
declare const KioskBoard: any;

@Component({
  selector: 'app-virtual-keyboard',
  template: `<input [type]="type" [placeholder]="placeholder" class="kiosk-input">`,
  styleUrls: ['./virtual-keyboard.component.scss']
})
export class VirtualKeyboardComponent implements AfterViewInit, OnDestroy {
  @Input() layout: 'alpha' | 'numeric' = 'alpha';
  @Input() placeholder = '';
  @Input() type = 'text';

  private inputEl!: HTMLInputElement;
  private focusListener?: () => void;
  private blurListener?: () => void;
  private keyboardVisible = false;

  constructor(private el: ElementRef, private renderer: Renderer2) {}

  ngAfterViewInit() {
    this.inputEl = this.el.nativeElement.querySelector('input');

    // listener per focus
    this.focusListener = this.renderer.listen(this.inputEl, 'focus', () => {
      if (!this.keyboardVisible) {
        this.openKeyboard();
        this.keyboardVisible = true;
      }
    });

    // listener per blur
    this.blurListener = this.renderer.listen(this.inputEl, 'blur', () => {
      setTimeout(() => this.closeKeyboard(), 200);
    });
  }

  private openKeyboard() {
    const config = {
      language: 'it',
      theme: 'light',
      allowRealKeyboard: false,
      allowMobileKeyboard: false,
      cssAnimations: true,
      cssAnimationsDuration: 360,
      cssAnimationsStyle: 'slide-up',
    };

    if (this.layout === 'alpha') {
      KioskBoard.run(this.inputEl, {
        ...config,
        keysArrayOfObjects: [
          { "0": "q", "1": "w", "2": "e", "3": "r", "4": "t", "5": "y", "6": "u", "7": "i", "8": "o", "9": "p" },
          { "0": "a", "1": "s", "2": "d", "3": "f", "4": "g", "5": "h", "6": "j", "7": "k", "8": "l", "9": "à", "10": "ò", "11": "ù", "12": "è" },
          { "0": "z", "1": "x", "2": "c", "3": "v", "4": "b", "5": "n", "6": "m", "7": ",", "8": ".", "9": "-" }
        ]
      });
    } else {
      KioskBoard.run(this.inputEl, {
        ...config,
        keysArrayOfObjects: [
          { "0": "1", "1": "2", "2": "3" },
          { "0": "4", "1": "5", "2": "6" },
          { "0": "7", "1": "8", "2": "9" },
          { "0": ",", "1": "0", "2": "{backspace}" }
        ]
      });
    }
  }

  private closeKeyboard() {
    const kb = document.querySelector('.kioskboard-container');
    if (kb) kb.remove();
    this.keyboardVisible = false;
  }

  ngOnDestroy() {
    this.focusListener?.();
    this.blurListener?.();
    this.closeKeyboard();
  }
}
