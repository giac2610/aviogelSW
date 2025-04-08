import { Directive, EventEmitter, HostListener, Output } from '@angular/core';

@Directive({
  selector: '[appLongPress]',
  standalone: false,
})
export class LongPressDirective {
  @Output() longPress = new EventEmitter<void>();
  private pressTimeout: any;

  @HostListener('touchstart', ['$event'])
  onTouchStart(event: TouchEvent): void {
    this.pressTimeout = setTimeout(() => {
      this.longPress.emit();
    }, 500); // Durata della pressione (500ms)
  }

  @HostListener('touchend')
  @HostListener('touchcancel')
  onTouchEnd(): void {
    clearTimeout(this.pressTimeout);
  }
}